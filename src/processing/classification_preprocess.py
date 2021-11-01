# creating dataset by cropping annotations from images
import yaml
import os
import random
from PIL import Image
from tqdm import tqdm
from xml.etree import ElementTree as ET
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

def get_config(file = "cpu_config"):
    """
    Get the config from the specified config file.
    
    Parameters:
    file (str): Name of the config file
    
    Returns:
    obj: yaml object    
    """

    # project_path = ""
    # config_path = os.path.join(project_path, 
    #                             "config/" + file + ".yaml")

    config_path = 'crop.yaml'
    with open(config_path) as file:
        config = yaml.full_load(file)
        
    return config 

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
    
class PrepareCropImages:
    
    def __init__(self):
        self.config = get_config("similarity_config")
        self.data_folder = self.config['image_crop']['data_folder']
        self.all_images_path = self.config['image_crop']['all_images_path']
        self.all_annotations_path = self.config['image_crop']['all_annotations_path']
        self.cropped_images_path = self.config['image_crop']['cropped_images_path']
        self.threshold = self.config['image_crop']['threshold']
        self.image_size = self.config['image_crop']['image_size']
        self.valid_size = self.config['image_crop']['valid_size']
        
        self.all_annotations = os.listdir(self.data_folder + self.all_annotations_path)
        random.shuffle(self.all_annotations)
        
        self.all_images = os.listdir(self.data_folder + self.all_images_path)
        random.shuffle(self.all_images)
        
        n_training = int(len(self.all_annotations) * (1-self.valid_size))
        
        self.training_annotations = self.all_annotations[0:n_training]
        self.validation_annotations = self.all_annotations[n_training:]
        
        self.image_label_dict_train = {}
        self.image_label_dict_valid = {}
        self.labels_crossed_threshold = []
       
    
    def crop_images(self):
        for image_name in tqdm(self.all_images):
            image_path = self.data_folder + self.all_images_path + image_name
            
            # annotation file and path
            annotation_name = '.'.join(image_name.split('.')[:-1]) + '.xml'
            if annotation_name not in self.all_annotations:
                continue
            
            annotation_path = self.data_folder + self.all_annotations_path + annotation_name
            
            # annotation
            image_annotation = read_annotation(annotation_path, use_lookup= True)
            
            image = Image.open(image_path)
            for i in image_annotation[1]:
                label = i['class']
                
                # threshold to be applied only on training dataset
                if annotation_name in self.training_annotations:
                    if label in self.labels_crossed_threshold:
                        continue
                    
                image_cropped = image.crop((i['xmin'], i['ymin'], i['xmax'], i['ymax']))
                
                if self.image_size:
                    image_cropped = image_cropped.resize(size= (self.image_size, self.image_size))
        
                if annotation_name in self.training_annotations:
                    if label in self.image_label_dict_train.keys():
                        self.image_label_dict_train[label].append(image_cropped)
                        
                        if self.threshold:
                            if len(self.image_label_dict_train[label]) >= self.threshold:
                                self.labels_crossed_threshold.append(label)
                    else:
                        self.image_label_dict_train[label] = [image_cropped]
                
                elif annotation_name in self.validation_annotations:
                    if label in self.image_label_dict_valid.keys():
                        self.image_label_dict_valid[label].append(image_cropped)
                    else:
                        self.image_label_dict_valid[label] = [image_cropped]
                else:
                    print('Annotation missing in both train and validation')
        
    
    def save_images(self, label_dict, sub_folder):

        from datetime import date
        today = date.today()
        date = today.strftime("%m%d")

        self.save_to_path = f"{self.data_folder}{self.cropped_images_path}{date}_{self.threshold}threshold_{self.image_size}image_size_{len(self.image_label_dict_train.keys())}folders"

        try:
            os.mkdir(self.save_to_path)
        except:
            pass

        self.save_to_path = f"{self.save_to_path}/{sub_folder}"

        try:
            os.mkdir(self.save_to_path)
        except:
            pass


        for i in label_dict.keys():
            try:
                curr_folder = self.save_to_path + "/" + i
                os.mkdir(curr_folder)
            except:
                pass

            for c, image in enumerate(label_dict[i]):
                image_name = 'Image ' + str(c+1)
                try:
                    image.save(f"{curr_folder}/{image_name}.jpg")
                except:
                    print('Issue with image')
                    
        if sub_folder == 'valid':
            for i in self.image_label_dict_train.keys():
                if i not in self.image_label_dict_valid.keys():
                    try:
                        curr_folder = self.save_to_path + "/" + i
                        os.mkdir(curr_folder)
                        print('Created dummy folder' + curr_folder)
                    except:
                        pass
                    
                
    def crop_and_save(self):
        self.crop_images()
        
        self.save_images(self.image_label_dict_train, 'train')
        self.save_images(self.image_label_dict_valid, 'valid')