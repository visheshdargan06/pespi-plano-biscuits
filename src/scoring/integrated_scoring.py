#TODO: implement integrated scoring
#  import yolov5 class
# import classification model
# import utils that load images and work on json (scoring loaders)
import os
import sys
import json
import pickle

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.scoring_loaders import image_loader, intermediate_json, final_json
from utils.config import get_config
from tensorflow.keras import models
import numpy as np 

#from detect_modified import Detect_YOLOv5
#import torch

import gc
gc.collect()
#torch.cuda.empty_cache()

class IntegratedScoring:
    def __init__(self, curr_model, index_lower=None, index_upper=None):
        self.curr_model = curr_model
        self.index_lower = index_lower
        self.index_upper = index_upper
        #self.config = get_config()
        self.config = get_config("production_config")
        self.blob_base_dir = self.config['data_path']['blob_base_dir']
        
        ###### productionise changes required for 2 folder ########
        #self.images_dir = self.blob_base_dir + self.config['data_path']['images_dir']
        self.output_images_folder = self.config['data_path']['output_images_folder']

        if self.curr_model in ['packets_detection', 'rackrow_detection']:  
            self.output_dir = self.blob_base_dir + self.config[curr_model]['output_dir'] + self.output_images_folder
            try:
                os.mkdir(self.output_dir)
            except:
                pass
            #self.image_size = self.config[curr_model]['image_size']

        elif self.curr_model == 'sub_brand_detection':
            self.model_dir = self.blob_base_dir + self.config[curr_model]['model_dir']
            self.output_dir = self.blob_base_dir + self.config[curr_model]['output_dir'] + self.output_images_folder
            self.input_json = self.blob_base_dir + self.config[curr_model]['input_json'] + self.output_images_folder
            self.cropped_size = int(self.config[curr_model]['cropped_size'])
            
            try:
                os.mkdir(self.output_dir)
            except:
                pass

            self.model = models.load_model(self.model_dir + 'model.h5')
            
            with open(self.model_dir + "class_indices.pickle", 'rb') as handle:
                self.class_indices_dict = pickle.load(handle)
            self.class_indices_dict = {value:key for key,value in self.class_indices_dict.items()}

    def predictions_classification(self, save= True):
        '''scoring and saving of the classification models like InceptionResNet-V2'''
        
        for i_loader in image_loader(index_lower= self.index_lower, index_upper= self.index_upper):
            image_name = i_loader[0]
            image = i_loader[1]
            width, height = i_loader[2][0], i_loader[2][1]
            
            json_file = ".".join(image_name.split(".")[:-1]) + '.json'
            with open(self.input_json + json_file, 'r') as json_file:
                i = json.load(json_file)
                
            i['image_dimension'] = (width, height)
                
            # cropping and predicting using classification model
            for j in i['packets']:
                image_cropped = image.crop([float(j['x1']), float(j['y1']), float(j['x2']), float(j['y2'])])
                image_cropped = image_cropped.resize(size= (self.cropped_size, self.cropped_size))
                
                image_cropped = np.array(image_cropped)/255.0
                image_cropped = image_cropped.reshape(1,self.cropped_size,self.cropped_size,3)
                
                pred_label = self.class_indices_dict[np.argmax(self.model.predict(image_cropped))]
                j['sub_brand'] = pred_label
                
            file_name = ".".join(image_name.split(".")[:-1]) + ".json"
            file_name = self.output_dir + file_name
            
            intermediate_json(i, file_path=file_name)

    def predictions_yolo(self):
        '''scoring of the yolo models'''
        with torch.no_grad():
            model_type = self.curr_model.split('_')[0]
            detect = Detect_YOLOv5(model_type=model_type)
            self.results = detect.main()

    def save_image_json_yolo(self):
        '''Saving the model output as JSON files (only for YOLO models)
        
            Output Structure:
                {'image_name':abc.jpg,
                'row_boxes' : [{'x1':10, 'y1':10, 'x2':20, 'y2':20}, {'x1':10, 'y1':10, 'x2':20, 'y2':20}, ...],
                'packets':[{'sub_brand':'Sabritas||Original', 'x1':10, 'y1':10, 'x2':20, 'y2':20},
                            {'sub_brand':'Doritos||Nachos', 'x1':30, 'y1':30, 'x2':40, 'y2':40},
                            ....]}
        '''
            
        for image_result in self.results:
            if self.curr_model == 'packets_detection':
                for i in range(0, len(image_result['packets'])):
                        image_result['packets'][i]["x1"] = str(image_result['packets'][i]['x1'])
                        image_result['packets'][i]["y1"] = str(image_result['packets'][i]['y1'])
                        image_result['packets'][i]["x2"] = str(image_result['packets'][i]['x2'])
                        image_result['packets'][i]["y2"] = str(image_result['packets'][i]['y2'])

            elif self.curr_model == 'rackrow_detection':
                image_result['row_boxes'] = image_result['rackrow']
                del image_result['rackrow']
                for i in range(0, len(image_result['row_boxes'])):
                        image_result['row_boxes'][i]["x1"] = str(image_result['row_boxes'][i]['x1'])
                        image_result['row_boxes'][i]["y1"] = str(image_result['row_boxes'][i]['y1'])
                        image_result['row_boxes'][i]["x2"] = str(image_result['row_boxes'][i]['x2'])
                        image_result['row_boxes'][i]["y2"] = str(image_result['row_boxes'][i]['y2'])

            else:
                pass

            img_name = image_result['image_name']
            file_name = ".".join(img_name.split(".")[:-1]) + ".json"
            file_name = self.output_dir + file_name
            
            intermediate_json(image_result, file_path=file_name)

    def integrate_model_outputs(self):
        '''Merges the sub-brand output with rack row output'''
        
        self.config = get_config("production_config")
        
        self.sub_brand_output_dir = self.blob_base_dir + self.config['integrated_output']['sub_brand_output_dir'] + self.output_images_folder
        self.rackrow_output_dir = self.blob_base_dir + self.config['integrated_output']['rackrow_output_dir'] + self.output_images_folder
        
        
        for json_output in os.listdir(self.sub_brand_output_dir):
            if 'ipynb' in json_output:
                continue
            
            with open(self.sub_brand_output_dir + json_output, 'r') as json_file:
                sub_brand_output = json.load(json_file)
                
            with open(self.rackrow_output_dir + json_output, 'r') as json_file:
                rack_row_output = json.load(json_file)
        
            sub_brand_output['row_boxes'] = rack_row_output['rackrow']
            sub_brand_output['image_dir'] = self.images_dir
            
            final_json(sub_brand_output, json_output)

    def scoring(self):
        '''Final Scoring of the images (includes all the models and their integrations)'''
        
        if self.curr_model in ['packets_detection', 'rackrow_detection']:
            self.predictions_yolo()
            self.save_image_json_yolo()
            
        elif self.curr_model == 'sub_brand_detection':
            self.predictions_classification()
            
        elif self.curr_model == 'integration':
            self.integrate_model_outputs()
            
        else:
            pass