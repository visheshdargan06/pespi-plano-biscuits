import json
import xml.etree.ElementTree as ET
from torch._C import dtype
import xmltodict
import numpy as np
import torch
from torchvision.ops import box_iou
import os
from tqdm import tqdm

def get_iou_per_image(pred_file, truth_file):

    with open(pred_file) as f:
        pred_json = json.load(f)
    pred_boxes = []
    for i in pred_json['packets']:
        pred_boxes.append([i['x1'], i['y1'], i['x2'], i['y2']])

    truth_boxes = []
    tree = ET.parse(truth_file)
    xml_data = tree.getroot()
    #here you can change the encoding type to be able to set it to the one you need
    xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')

    data_dict = dict(xmltodict.parse(xmlstr))
    for i in data_dict['annotation']['object']:
        truth_boxes.append(list(i['bndbox'].values()))

    # print("Total true boxes: ", len(truth_boxes))
    # print("Total predicted boxes: ", len(pred_boxes))

    pred_boxes = np.array(pred_boxes).astype('int')
    truth_boxes = np.array(truth_boxes).astype('int')
    
    
    pred_boxes = torch.from_numpy(pred_boxes).float()
    truth_boxes = torch.from_numpy(truth_boxes).float()

    try:
        box_iou_all, detection_score = box_iou(pred_boxes, truth_boxes).numpy(), len(pred_boxes)*2/(len(truth_boxes) + len(pred_boxes))
    except:
        box_iou_all, detection_score = 0, 0
    return box_iou_all, detection_score



def get_iou(pred_files, truth_files):
    '''
    pred_files: list of predicted files
    truth_files: list of truth files

    Returns: dict of iou per image each dict key is the image name, each value is dict with best_iou and detection score
    '''

    iou_per_image = dict()

    # normalised detection score
    # detected * 2 / (detected + actual)


    for i in os.walk(pred_files):
        # print(i[2])
        for j in tqdm(i[2]):
            iou_matrix, detection_score  = get_iou_per_image(pred_files+j, truth_files+j.split('.')[0]+'.xml')
            #print(iou_matrix,detection_score)
            try:
                best_iou = iou_matrix.max(0)
            except:
                best_iou = np.array([])

            iou_per_image[j.split('.')[0]] = {'best_iou' : best_iou.tolist(), 'detection_score': detection_score}

    return iou_per_image

if __name__=="__main__":
    pred_files = '/media/premium/common-biscuit/main/planogram_biscuit/data/output/image_annotations/packets_detection/op_annotations_medium/'
    truth_files = '/media/premium/common-biscuit/main/planogram_biscuit/data/raw/annotations_master/'
    iou_per_image = get_iou(pred_files,truth_files)
    print(iou_per_image)
    with open('/media/premium/common-biscuit/main/planogram_biscuit/data/iou_json/packets_json/yolov5_medium_iou_file.json', 'w') as f:
	    json.dump(iou_per_image, f)

    