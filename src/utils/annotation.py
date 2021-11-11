import sys
sys.path.append("../")
from xml.etree import ElementTree as ET
from utils.config import get_config
import pandas as pd


def get_class(label):
    config = get_config()
    lookup = pd.read_csv(config['path']['packet_lookup'])
    lookup = lookup.dropna()
    try:
        label = lookup.loc[lookup['label_annotation_file']==label, 'display_label'].values[0]
    except:
        label = None
    
    return label

def read_annotation(file, exclude=[], use_lookup=True):
    
    tree = ET.parse(file)
    root_element = tree.getroot()
    image = root_element.find('filename').text
    bbox = []
    
    for obj in root_element.findall('object'):
        name = obj.find('name').text
        if name in exclude:
            continue
        if use_lookup:
            name = get_class(name)
            if name is None:
                continue
        for boxes in obj.findall('bndbox'):
            left = int(boxes.find('xmin').text)
            top = int(boxes.find('ymin').text)
            right = int(boxes.find('xmax').text)
            bottom = int(boxes.find('ymax').text)
            box = {"class": name,
                   "xmin": left,
                   "ymin": top,
                   "xmax": right,
                   "ymax": bottom}
            bbox.append(box)
            
    return image, bbox