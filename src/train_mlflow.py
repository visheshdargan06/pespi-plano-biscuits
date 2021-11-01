
import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.metrics import fitness
from utils.callbacks import Callbacks
from utils.config import get_config
from utils.mlflow import MLFlow


LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class Train_YOLOv5:
    def __init__(self, model_type):
        self.model_type = model_type
        if self.model_type == 'rackrow':
            self.config = get_config("YOLOv5_rackrow")
        elif self.model_type == 'packets':
            self.config = get_config("YOLOv5_packets")
        self.weights = self.config['yolov5_train']['weights']  
        self.cfg = self.config['yolov5_train']['cfg']
        self.data = self.config['yolov5_train']['data']
        self.hyp = self.config['yolov5_train']['hyp']
        self.epochs = self.config['yolov5_train']['epochs']
        self.batch_size = self.config['yolov5_train']['batch_size']
        self.imgsz = self.config['yolov5_train']['imgsz']
        self.resume = self.config['yolov5_train']['resume']
        self.rect = self.config['yolov5_train']['rect']
        self.nosave = self.config['yolov5_train']['nosave']
        self.noval = self.config['yolov5_train']['noval']
        self.noautoanchor = self.config['yolov5_train']['noautoanchor'] 
        self.evolve = self.config['yolov5_train']['evolve']   
        self.bucket = self.config['yolov5_train']['bucket']  
        self.cache = self.config['yolov5_train']['cache']
        self.image_weights = self.config['yolov5_train']['image_weights']
        self.device = self.config['yolov5_train']['device']
        self.multi_scale = self.config['yolov5_train']['multi_scale']
        self.single_cls = self.config['yolov5_train']['single_cls']
        self.adam = self.config['yolov5_train']['adam']
        self.sync_bn = self.config['yolov5_train']['sync_bn']

        self.workers = self.config['yolov5_train']['workers']
        self.project = self.config['yolov5_train']['project']
        self.name = self.config['yolov5_train']['name']
        self.exist_ok = self.config['yolov5_train']['exist_ok']
        self.quad = self.config['yolov5_train']['quad']
        self.linear_lr = self.config['yolov5_train']['linear_lr']
        self.label_smoothing = self.config['yolov5_train']['label_smoothing']

        self.patience = self.config['yolov5_train']['patience']
        self.freeze = self.config['yolov5_train']['freeze']
        self.save_period = self.config['yolov5_train']['save_period']
        self.local_rank = self.config['yolov5_train']['local_rank']
        
        self.entity = self.config['yolov5_train']['entity']
        self.upload_dataset = self.config['yolov5_train']['upload_dataset']
        self.bbox_interval = self.config['yolov5_train']['bbox_interval']
        self.artifact_alias = self.config['yolov5_train']['artifact_alias']
        
        

    def train(self, hyp,  device, callbacks):
        save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
            Path(self.save_dir), self.epochs, self.batch_size, self.weights, self.single_cls, self.evolve, self.data, self.cfg, \
            self.resume, self.noval, self.nosave, self.workers, self.freeze

        # Directories
        w = save_dir / 'weights'  # weights dir
        (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
        last, best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

        

        # Save run settings
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        # with open(save_dir / 'opt.yaml', 'w') as f:
        #     yaml.safe_dump(vars(opt), f, sort_keys=False)
        data_dict = None

        # # Loggers
        # if RANK in [-1, 0]:
        #     loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        #     if loggers.wandb:
        #         data_dict = loggers.wandb.data_dict
        #         if resume:
        #             weights, epochs, hyp = self.weights, self.epochs, self.hyp

        #     # Register actions
        #     for k in methods(loggers):
        #         callbacks.register_action(k, callback=getattr(loggers, k))

        # Config
        plots = not evolve  # create plots
        cuda = device.type != 'cpu'
        init_seeds(1 + RANK)
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = data_dict or check_dataset(data)  # check if None
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
        is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

        # Model
        check_suffix(weights, '.pt')  # check weights
        pretrained = weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
            exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        else:
            model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

        # Freeze
        freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        if self.adam:
            optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                    f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
        del g0, g1, g2

        # Scheduler
        if self.linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        else:
            lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        ema = ModelEMA(model) if RANK in [-1, 0] else None

        # Resume
        start_epoch, best_fitness = 0, 0.0
        if pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            if resume:
                assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
            if epochs < start_epoch:
                LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, csd

        # Image sizes
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz = check_img_size(self.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # DP mode
        if cuda and RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                            'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if self.sync_bn and cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                                hyp=hyp, augment=True, cache=self.cache, rect=self.rect, rank=LOCAL_RANK,
                                                workers=workers, image_weights=self.image_weights, quad=self.quad,
                                                prefix=colorstr('train: '))
        mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
        nb = len(train_loader)  # number of batches
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in [-1, 0]:
            val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                        hyp=hyp, cache=None if noval else self.cache, rect=True, rank=-1,
                                        workers=workers, pad=0.5,
                                        prefix=colorstr('val: '))[0]

            if not resume:
                labels = np.concatenate(dataset.labels, 0)
                # c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                if plots:
                    plot_labels(labels, names, save_dir)

                # Anchors
                if not self.noautoanchor:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end')

        # DDP mode
        if cuda and RANK != -1:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

        # Model parameters
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        hyp['label_smoothing'] = self.label_smoothing
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=cuda)
        stopper = EarlyStopping(patience=self.patience)
        compute_loss = ComputeLoss(model)  # init loss class
        LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {train_loader.num_workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', save_dir)}\n"
                    f'Starting training for {epochs} epochs...')
        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            model.train()

            # Update image weights (optional, single-GPU only)
            if self.image_weights:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(3, device=device)  # mean losses
            if RANK != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(train_loader)
            LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
            if RANK in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
            optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                if self.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if self.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni - last_opt_step >= accumulate:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    last_opt_step = ni

                # Log
                if RANK in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                        f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                    callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, self.sync_bn)
                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            scheduler.step()

            if RANK in [-1, 0]:
                # mAP
                callbacks.run('on_train_epoch_end', epoch=epoch)
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
                if not noval or final_epoch:  # Calculate mAP
                    results, maps, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=ema.ema,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            plots=False,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi
                log_vals = list(mloss) + list(results) + lr
                callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

                

                

                # Save model
                if (not nosave) or (final_epoch and not evolve):  # if save
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'model': deepcopy(de_parallel(model)).half(),
                            'ema': deepcopy(ema.ema).half(),
                            'updates': ema.updates,
                            'optimizer': optimizer.state_dict()}

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if best_fitness == fi:
                        torch.save(ckpt, best)
                    if (epoch > 0) and (self.save_period > 0) and (epoch % self.save_period == 0):
                        torch.save(ckpt, w / f'epoch{epoch}.pt')
                    del ckpt
                    callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

                # Stop Single-GPU
                if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                    break

                # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
                # stop = stopper(epoch=epoch, fitness=fi)
                # if RANK == 0:
                #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

            # Stop DPP
            # with torch_distributed_zero_first(RANK):
            # if stop:
            #    break  # must break all DDP ranks

            # end epoch ----------------------------------------------------------------------------------------------------
        # end training -----------------------------------------------------------------------------------------------------
        if RANK in [-1, 0]:
            LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
                    if f is best:
                        LOGGER.info(f'\nValidating {f}...')
                        results, _, _ = val.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                model=attempt_load(f, device).half(),
                                                iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                save_json=is_coco,
                                                verbose=True,
                                                plots=True,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)  # val best model with plots

            callbacks.run('on_train_end', last, best, plots, epoch)
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

        torch.cuda.empty_cache()
        #mlflow logging
        best_metrics = {'P R mAP_.5 mAP_.5_.95 val_loss_box val_loss_obj val_loss_cls':list(results),  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                        'epoch':epoch,
                        'best_fitness':best_fitness,
                        'fi':fi}
        self.mlflow_obj.log_params(best_metrics)
        return results


    def main(self, callbacks=Callbacks()):
        # Checks
        set_logging(RANK)
        if RANK in [-1, 0]:
            check_git_status()
            check_requirements(exclude=['thop'])

        # Resume
        if self.resume and not self.evolve:  # resume an interrupted run
            ckpt = self.resume if isinstance(self.resume, str) else get_latest_run()  # specified or most recent path
            assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
            with open(Path(ckpt).parent.parent / 'self.yaml', errors='ignore') as f:
                opt = argparse.Namespace(**yaml.safe_load(f))  # replace
            self.cfg, self.weights, self.resume = '', ckpt, True  # reinstate
            LOGGER.info(f'Resuming training from {ckpt}')
        else:
            self.data, self.cfg, self.hyp, self.weights, self.project = \
                check_file(self.data), check_yaml(self.cfg), check_yaml(self.hyp), str(self.weights), str(self.project)  # checks
            assert len(self.cfg) or len(self.weights), 'either --cfg or --weights must be specified'
            if self.evolve:
                self.project = str(ROOT / 'runs/evolve')
                self.exist_ok, self.resume = self.resume, False  # pass resume to exist_ok and disable resume
            self.save_dir = str(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))

        # DDP mode
        device = select_device(self.device, batch_size=self.batch_size)
        if LOCAL_RANK != -1:
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            assert self.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
            assert not self.image_weights, '--image-weights argument is not compatible with DDP training'
            assert not self.evolve, '--evolve argument is not compatible with DDP training'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # Train
        if not self.evolve:
            self.train(self.hyp, device, callbacks)
            if WORLD_SIZE > 1 and RANK == 0:
                LOGGER.info('Destroying process group... ')
                dist.destroy_process_group()

        # Evolve hyperparameters (optional)
        else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
            meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                    'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                    'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                    'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                    'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                    'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                    'box': (1, 0.02, 0.2),  # box loss gain
                    'cls': (1, 0.2, 4.0),  # cls loss gain
                    'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                    'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                    'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                    'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                    'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                    'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                    'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                    'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                    'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                    'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                    'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                    'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                    'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                    'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                    'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                    'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                    'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                    'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                    'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

            with open(self.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
                if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                    hyp['anchors'] = 3
            self.noval, self.nosave, save_dir = True, True, Path(self.save_dir)  # only val/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
            if self.bucket:
                os.system(f'gsutil cp gs://{self.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

            for _ in range(self.evolve):  # generations to evolve
                if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.8, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                    ng = len(meta)
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

                # Constrain to limits
                for k, v in meta.items():
                    hyp[k] = max(hyp[k], v[1])  # lower limit
                    hyp[k] = min(hyp[k], v[2])  # upper limit
                    hyp[k] = round(hyp[k], 5)  # significant digits

                # Train mutation
                results = self.train(hyp.copy(), device, callbacks)

                # Write mutation results
                print_mutation(results, hyp.copy(), save_dir, self.bucket)

            # Plot results
            plot_evolve(evolve_csv)
            print(f'Hyperparameter evolution finished\n'
                f"Results saved to {colorstr('bold', save_dir)}\n"
                f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


    def run(self):
        # starting and closing ML flow session
        artifacts_location = self.config['mlflow_logging']['artifacts_location']
        uri_train_valid = self.config['mlflow_logging']['uri_train_valid']
        self.temp_artifacts_location = self.config['mlflow_logging']['temp_artifacts_location']
        self.mlflow_obj = MLFlow(experiment_name= self.model_type,
                                 artifacts_location= artifacts_location, tracking_uri= uri_train_valid, 
                                 temp_artifacts_folder= self.temp_artifacts_location,
                                 delete_existing= True)
        self.mlflow_obj.define_session()
        self.main()
        #self.log_mlflow_params()
        self.mlflow_obj.log_artifact(delete_existing= True)
        self.mlflow_obj.end_session()


        


if __name__ == "__main__":
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()



    t = Train_YOLOv5(model_type = 'packets')
    #t.main()
    t.run()