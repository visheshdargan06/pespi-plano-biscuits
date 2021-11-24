
import os
import sys
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
    
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#print(sys.path)
#print(os.getcwd())   

from utils.config import get_config

class YOLODataPrep:
    '''
        prepares the data to be consumed by YOLO and other models
        converts PASCAL VOC labels to YOLO format
        creates annotation file images
    '''
    
    def __init__(self, model_type=None):
        '''
        model_type: either of packets, sub_brands_without_others, sub_brands_with_others, complete_rack, rack_row
        '''
        self.model_type = model_type
        self.config = get_config("production_config_main")
        
        self.data_folder = self.config['path']['blob_base_dir'] + self.config['path']['data']
        self.validation_folder = self.config['path']['validation_images']  # to skip these images
        self.images_folder = self.data_folder + "raw/all_images_12Nov/"
        self.labels_folder = self.data_folder + "raw/annotations_master_642/"
        
        self.lookup_file = self.data_folder + self.config['path'][self.model_type+'_lookup']
        
        self.class_dict_modified = pd.read_csv(self.lookup_file,
                                               index_col= 'label_annotation_file')['label_index'].dropna().to_dict()
        self.class_dict_modified = {key:int(value) for key, value in self.class_dict_modified.items() if value >= 0}
    
    def convert(self, size, box):
        '''
        conversion of PASCAL VOC to YOLO format
        '''
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1
        y = (box[2] + box[3])/2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)
            
    def prepare_data(self, split = {'train':0.7, 'validate':0.2, 'test':0.1}):
        '''
        prepares and store the labelled file in yolo format
        '''
        images = os.listdir(self.labels_folder)
        if 'validate' in split:
            train, validate, test = np.split(images, [int(len(images)*split['train']), 
                                                    int(len(images)*(split['train']+split['validate']))])
            print(len(train), len(validate),len(test))
        else:
            train, validate, test = np.split(images, [int(len(images)*split['train']), 
                                                    int(len(images))])
            print(len(train), len(validate),len(test))


        if len(train)>0:
            for train_file in tqdm(train):
                label_file = train_file.split(".")[:-1]
                train_image_file = '.'.join(label_file) + '.jpg'
                label_file_txt = '.'.join(label_file) + ".txt"
                label_file = '.'.join(label_file) + ".xml"
                # annotated images
                if label_file in os.listdir(self.labels_folder):
                    single_label_file = open(self.data_folder + self.model_type + '_dataset/labels/train/' + label_file_txt, 'w')

                    # saving images to respective folder
                    train_image_folder = self.data_folder + self.model_type + '_dataset/images/train/'
                    if train_image_file not in os.listdir(train_image_folder):
                        i = Image.open(self.images_folder + train_image_file)
                        #i_bw = i.convert('1').copy()
                        #i_bw.save(self.data_folder + 'processed/bw_images/' + image_file)
                        i.save(train_image_folder + train_image_file)
                    
                    # preparing the labels file
                    label_file_path = self.labels_folder + label_file
                    tree = ET.parse(label_file_path)
                    root = tree.getroot()
                    size = [int(root.find('size').find('width').text), int(root.find('size').find('height').text)]
                    
                    for obj in root.iter('object'):
                        cls = obj.find('name').text

                        if cls not in self.class_dict_modified.keys():
                            continue
                            
                        xmlbox = obj.find('bndbox')
                        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                        b = self.convert(size,b)
                        
                        cls_id = str(self.class_dict_modified[cls])

                        row = cls_id +" "+ str(round(b[0], 5)) + " " + str(round(b[1], 5)) + " " + str(round(b[2], 5)) + " " + str(round(b[3], 5))
                        single_label_file.write(row + '\n')
                    single_label_file.close()

        if len(validate)>0:
            for val_file in tqdm(validate):
                label_file = val_file.split(".")[:-1]
                val_image_file = '.'.join(label_file) + '.jpg'
                label_file_txt = '.'.join(label_file) + ".txt"
                label_file = '.'.join(label_file) + ".xml"
                # annotated images
                if label_file in os.listdir(self.labels_folder):
                    single_label_file = open(self.data_folder + self.model_type + '_dataset/labels/val/' + label_file_txt, 'w')

                    # saving images to respective folder
                    val_image_folder = self.data_folder + self.model_type + '_dataset/images/val/'
                    if val_image_file not in os.listdir(val_image_folder):
                        i = Image.open(self.images_folder + val_image_file)
                        #i_bw = i.convert('1').copy()
                        #i_bw.save(self.data_folder + 'processed/bw_images/' + image_file)
                        i.save(val_image_folder + val_image_file)
                    
                    # preparing the labels file
                    label_file_path = self.labels_folder + label_file
                    tree = ET.parse(label_file_path)
                    root = tree.getroot()
                    size = [int(root.find('size').find('width').text), int(root.find('size').find('height').text)]
                    
                    for obj in root.iter('object'):
                        cls = obj.find('name').text

                        if cls not in self.class_dict_modified.keys():
                            continue
                            
                        xmlbox = obj.find('bndbox')
                        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                        b = self.convert(size,b)
                        
                        cls_id = str(self.class_dict_modified[cls])

                        row = cls_id +" "+ str(round(b[0], 5)) + " " + str(round(b[1], 5)) + " " + str(round(b[2], 5)) + " " + str(round(b[3], 5))
                        single_label_file.write(row + '\n')
                    single_label_file.close()

        if len(test)>0:
            for test_file in tqdm(test):
                label_file = test_file.split(".")[:-1]
                test_image_file = '.'.join(label_file) + '.jpg'
                label_file_txt = '.'.join(label_file) + ".txt"
                label_file = '.'.join(label_file) + ".xml"
                # annotated images
                if label_file in os.listdir(self.labels_folder):
                    single_label_file = open(self.data_folder + self.model_type + '_dataset/labels/test/' + label_file_txt, 'w')

                    # saving images to respective folder
                    test_image_folder = self.data_folder + self.model_type + '_dataset/images/test/'
                    if test_image_file not in os.listdir(test_image_folder):
                        i = Image.open(self.images_folder + test_image_file)
                        #i_bw = i.convert('1').copy()
                        #i_bw.save(self.data_folder + 'processed/bw_images/' + image_file)
                        i.save(test_image_folder + test_image_file)
                    
                    # preparing the labels file
                    label_file_path = self.labels_folder + label_file
                    tree = ET.parse(label_file_path)
                    root = tree.getroot()
                    size = [int(root.find('size').find('width').text), int(root.find('size').find('height').text)]
                    
                    for obj in root.iter('object'):
                        cls = obj.find('name').text

                        if cls not in self.class_dict_modified.keys():
                            continue
                            
                        xmlbox = obj.find('bndbox')
                        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                        b = self.convert(size,b)
                        
                        cls_id = str(self.class_dict_modified[cls])

                        row = cls_id +" "+ str(round(b[0], 5)) + " " + str(round(b[1], 5)) + " " + str(round(b[2], 5)) + " " + str(round(b[3], 5))
                        single_label_file.write(row + '\n')
                    single_label_file.close()    
        

if __name__ == "__main__":
    data_prep = YOLODataPrep(model_type='rackrow')
    data_prep.prepare_data(split = {'train':1})
    print(data_prep.class_dict_modified)