import json
import xml.etree.ElementTree as ET
import xmltodict
import numpy
import torch
from torchvision.ops import box_iou


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

    print("Total true boxes: ", len(truth_boxes))
    print("Total predicted boxes: ", len(pred_boxes))

    pred_boxes = numpy.array(pred_boxes).astype('int')
    truth_boxes = numpy.array(truth_boxes).astype('int')
    
    
    pred_boxes = torch.from_numpy(pred_boxes).float()
    truth_boxes = torch.from_numpy(truth_boxes).float()

    return box_iou(pred_boxes, truth_boxes).numpy()



def get_iou(pred_files, truth_files):
    pass

iou_matrix = get_iou_per_image('/media/premium/common-biscuit/main/planogram_biscuit/data/output/image_annotations/packets_detection/op_annotations/PHOTO-2021-07-21-09-17-10.json', '/media/premium/common-biscuit/main/planogram_biscuit/data/raw/annotations_master/PHOTO-2021-07-21-09-17-10.xml')

best_iou = iou_matrix.max(0)
print(best_iou.shape)