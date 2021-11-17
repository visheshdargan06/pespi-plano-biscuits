import pathlib
import matplotlib
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import logging as logger
import json
import os
matplotlib.use('Agg')

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import json
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def save_final_json(image_name, json_result, json_path= None):
    
    from datetime import date
    today = date.today().strftime("_%d_%m_%Y")
    folder_name = "json_outputs_" + today
    json_path = os.path.join(json_path, folder_name)
                             
    json_file_name = image_name + ".json"
    pathlib.Path(json_path).mkdir(parents=True, exist_ok=True) 
    with open(os.path.join(json_path, json_file_name), 'w') as f:
        f.write(json.dumps(json_result, indent=2, separators=(',', ': ')))
        
        
def save_final_csv(df, root_path):
    pathlib.Path(root_path).mkdir(parents=True, exist_ok=True)
    
    from datetime import date
    today = date.today().strftime("_%d_%m_%Y")
    file_name = "detections.csv"
    
    df.to_csv(os.path.join(root_path, file_name), index= False)
    
    
def test_logger(text):
    logger.info(text)
    logger.warning(text)
    logger.error(text)


def save_object(obj, filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file)


def read_object(filename):
    return pickle.load(open(filename, 'rb'))
        

def save_csv(obj, filename, save_index=False):
    obj.to_csv(filename, index=save_index)


def read_data(filename):
    # TODO: Implement read function
    pass


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the \
               Decision Threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.ylim([0, 1])
    return plt


def plot_roc_curve(y_val_splt, predicted_prob):

    fpr, tpr, thresholds = roc_curve(y_val_splt, predicted_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.plot([0, 1], [0, 1], color='red',  linestyle='--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    return plt


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    
    image_name = ''
    for i in annotation_line.split():
        image_name = image_name + i + ' '
        if (i.endswith('.png') | i.endswith('.jpeg') | i.endswith('.jpg') | i.endswith('.PNG') | i.endswith('.JPEG') | i.endswith('.JPG')):
            break
    image_name = image_name[:-1]   
    image_line = annotation_line.replace(image_name, '')[1:].split()
    
    line = [image_name] + image_line

#     line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data




def visualize_integrated_output(json_result, visualization):
    '''Visualize the model outputs from the final integrated JSON results.
    Params:
        json_result: path of json file that has an integrated output of all the models
        visualization: packets/row_boxes/all (all refers to both packets and row_boxes)
    
    '''
    
    with open(json_result, 'r') as json_file:
        json_result = json.load(json_file)
        
    images_dir = json_result['image_dir']
    image = Image.open(images_dir + json_result['image_name'])
    
    font = ImageFont.truetype(font= '../../font/FiraMono-Medium.otf',
                          size= np.floor(0.01 * image.size[1] + 2).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 380
    
    if (visualization == 'packets') | (visualization == 'all'):
        for i in json_result['packets']:
            predicted_class = i['sub_brand']
            label = '{}'.format(predicted_class.replace('||', '\n'))
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
    
            left, top, right, bottom = int(i['x1']), int(i['y1']), int(i['x2']), int(i['y2'])
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    
            if top - label_size[1] >= 0:
                text_origin = np.array([left, (top + top + bottom)/3])
            else:
                text_origin = np.array([left, (top + top + bottom)/3])
    
            draw.rectangle([left, top, right, bottom], width= thickness)
            
            draw.multiline_text(text_origin, label, fill=(255, 255, 255), font=font)
                
    
    if (visualization == 'row_boxes') | (visualization == 'all'):
        for i in json_result['row_boxes']:
            predicted_class = 'Rack Row'
            label = '{}'.format(predicted_class.replace('||', '\n'))
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
    
            left, top, right, bottom = int(i['x1']), int(i['y1']), int(i['x2']), int(i['y2'])
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    
            if top - label_size[1] >= 0:
                text_origin = np.array([left, (top + top + bottom)/3])
            else:
                text_origin = np.array([left, (top + top + bottom)/3])
    
            draw.rectangle([left, top, right, bottom], width= thickness)
            
            draw.multiline_text(text_origin, label, fill=(255, 255, 255), font=font)
            
    del draw
    
    return image
