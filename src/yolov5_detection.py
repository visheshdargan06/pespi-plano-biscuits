import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory(src)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors, get_gradient_color
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.config import get_config

class Detect_YOLOv5:
    def __init__(self, model_type):
        self.model_type = model_type
        self.detect = get_config("production_config_main")
        self.config = self.detect["yolov5_detect_"+self.model_type]
        self.model_version = self.config['model_version']
        if self.model_version == 'small':
            self.weights = self.config['weights_small']  
        else:
            self.weights = self.config['weights_medium']
        self.source = self.detect['data_path']['blob_base_dir']+self.detect['data_path']['images_dir']
        self.imgsz = self.config['imgsz']
        self.conf_thres = self.config['conf_thres']
        self.iou_thres = self.config['iou_thres']
        self.max_det = self.config['max_det']
        self.device = self.config['device']
        self.view_img = self.config['view_img']
        self.save_txt = self.config['save_txt']
        self.save_conf = self.config['save_conf']
        self.save_crop = self.config['save_crop']
        self.nosave = self.config['nosave'] 
        self.classes = self.config['classes']   
        self.agnostic_nms = self.config['agnostic_nms']  
        self.augment = self.config['augment']
        self.visualize = self.config['visualize']
        self.update = self.config['update']
        self.project = self.config['project']
        self.name = self.config['name']
        self.exist_ok = self.config['exist_ok']
        self.line_thickness = self.config['line_thickness']
        self.hide_labels = self.config['hide_labels']
        self.hide_conf = self.config['hide_conf']
        self.half = self.config['half']


    @torch.no_grad()
    def run(self):
        source = str(self.source)
        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        if save_img or self.save_txt or self.visualize:
            save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
            (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        self.half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        if pt:
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.weights, map_location=device)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if self.half:
                model.half()  # to FP16
            if classify:  # second-stage classifier
                modelc = load_classifier(name='resnet50', n=2)  # initialize
                modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
        elif onnx:
            if self.dnn:
                check_requirements(('opencv-python>=4.5.4',))
                net = cv2.dnn.readNetFromONNX(w)
            else:
                check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
                import onnxruntime
                session = onnxruntime.InferenceSession(w, None)
        else:  # TensorFlow models
            check_requirements(('tensorflow>=2.4.1',))
            import tensorflow as tf
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                model = tf.keras.models.load_model(w)
            elif tflite:
                interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
                int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
        self.imgsz *= 2 if len(self.imgsz) == 1 else 1  # expand
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0

        output_img= []
        for path, img, im0s, vid_cap in dataset:
            img_out_dict = {"image_name":Path(path).name}

            t1 = time_sync()
            if onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            if pt:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred = model(img, augment=self.augment, visualize=visualize)[0]
            elif onnx:
                if self.dnn:
                    net.setInput(img)
                    pred = torch.tensor(net.forward())
                else:
                    pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
            else:  # tensorflow model (tflite, pb, saved_model)
                imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
                if pb:
                    pred = frozen_func(x=tf.constant(imn)).numpy()
                elif saved_model:
                    pred = model(imn, training=False).numpy()
                elif tflite:
                    if int8:
                        scale, zero_point = input_details[0]['quantization']
                        imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                    interpreter.set_tensor(input_details[0]['index'], imn)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    if int8:
                        scale, zero_point = output_details[0]['quantization']
                        pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
                pred[..., 0] *= imgsz[1]  # x
                pred[..., 1] *= imgsz[0]  # y
                pred[..., 2] *= imgsz[1]  # w
                pred[..., 3] *= imgsz[0]  # h
                pred = torch.tensor(pred)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            output_per_img = []
            complete_rack_detected = []
            confidence_per_img = []
            complete_rack_confidence = []
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                
                
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            

                            annotator.box_label(xyxy, label, color=get_gradient_color(int(conf * 100)))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        
                        c = int(cls)
                        label_updated = names[c]
                        if self.model_type == 'rackrow':
                            if label_updated == 'complete rack':
                                complete_rack_detected.append({ 
                                                    'x1': int(torch.tensor(xyxy).tolist()[0]), 
                                                    'y1': int(torch.tensor(xyxy).tolist()[1]), 
                                                    'x2': int(torch.tensor(xyxy).tolist()[2]), 
                                                    'y2': int(torch.tensor(xyxy).tolist()[3]) })
                                complete_rack_confidence.append(round(conf.item(),2))
                            else:
                                output_per_img.append({'x1': int(torch.tensor(xyxy).tolist()[0]), 
                                                        'y1': int(torch.tensor(xyxy).tolist()[1]), 
                                                        'x2': int(torch.tensor(xyxy).tolist()[2]), 
                                                        'y2': int(torch.tensor(xyxy).tolist()[3]) })
                                confidence_per_img.append(round(conf.item(),2))
                                                    
                        elif self.model_type == 'packets':
                            output_per_img.append({'sub_brand':label_updated, 
                                                    'x1': int(torch.tensor(xyxy).tolist()[0]), 
                                                    'y1': int(torch.tensor(xyxy).tolist()[1]), 
                                                    'x2': int(torch.tensor(xyxy).tolist()[2]), 
                                                    'y2': int(torch.tensor(xyxy).tolist()[3]) })
                            confidence_per_img.append(round(conf.item(),2))

                # Print time (inference-only)
                print(f'{s}Done. ({t3 - t2:.3f}s)')

                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    save_path = str(save_dir / p.name)  # img.jpg
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
            img_out_dict[self.model_type] = output_per_img

            if self.model_type == 'rackrow':
                img_out_dict['complete_rack'] = complete_rack_detected
                img_out_dict['complete_rack_confidence'] = complete_rack_confidence

            img_out_dict[self.model_type+"_confidence"] = confidence_per_img
            output_img.append(img_out_dict)


        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if self.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)
        
        return output_img

    def main(self):
        check_requirements(exclude=('tensorboard', 'thop'))
        output = self.run()
        return output


if __name__ == "__main__":
    detect = Detect_YOLOv5(model_type='rackrow')
    print(detect.main())
