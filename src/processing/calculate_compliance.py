import warnings;
warnings.filterwarnings('ignore');

import pickle
import os
import re
import sys
import math
import numpy as np
from collections import Counter
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from utils.utils import visualize_integrated_output, save_final_json, save_final_csv
from utils.config import get_config
from scoring.Locator import Locator
from utils.custom_datatypes import Packet, Row


def standardize_names(all_names, prev_after):
    '''creates a dataframe with original name and the corresponding name that can be later used for comparsion
        only the names works by extracting that have max of prev/after e.g prev, prev(1), prev(2)
    '''
    
    df = pd.DataFrame(columns = ['actual_name'])
    df['actual_name'] = all_names
    
    df['comparison_string'] = df.actual_name.apply(lambda x : x.split(prev_after)[0])
    df['remaining_string'] = df.actual_name.apply(lambda x : x.split(prev_after)[1])
    df['max_number'] = df.remaining_string.apply(lambda x : 0 if re.findall(r'\d+', x) == [] else re.findall(r'\d+', x)[0])
    df['max_number'] = df['max_number'].astype(int)
    
    df_latest_image = df.fillna(0).groupby('comparison_string')['max_number'].max().reset_index()
    df_latest_image = df.merge(df_latest_image)
    
    return df_latest_image[['actual_name', 'comparison_string']]


def create_position_df(output, flag):
    df = pd.DataFrame(columns= ['Sub-Brand', 'Position', flag])
    for idx_row in range(len(output)):
        rack_row = output[idx_row]
        
        for idx_col in range(len(rack_row)):
            sub_brand = rack_row[idx_col].label
            position = (idx_row+1, idx_col+1)
            df.loc[df.shape[0], :] = [sub_brand, position, 1]
    
    return df


def get_image_compliance(output_prev, output_after):
    '''get compliance by comparing outputs of before and previous images'''
    df_prev = create_position_df(output_prev, 'prev')
    df_after = create_position_df(output_after, 'after')
    
    df_merged = df_prev.merge(df_after, on= 'Position', how= 'outer')
    df_merged['row'] = df_merged['Position'].apply(lambda x : x[0])
    df_merged['column'] = df_merged['Position'].apply(lambda x : x[1])
    
    df_merged['Compliance'] = np.where(df_merged['Sub-Brand_x']==df_merged['Sub-Brand_y'], True, False)
        
    # calculation of overall compliance
    prev_sub_brands = df_merged['Sub-Brand_x'].values
    if len(prev_sub_brands) == 0:
        return {}
    after_sub_brands = df_merged['Sub-Brand_y'].values
    match = list((Counter(prev_sub_brands) & Counter(after_sub_brands)).elements())
    
    if len(prev_sub_brands) == 0:
        overall_rack_compliance = 0
    else:
        overall_rack_compliance = 100 * len(match)/len(prev_sub_brands)
    
    # calculation of row level compliance
    row_level_compliance = {}
    row_level_prev_count = {}
    row_level_after_count = {}
    
    for i in df_merged.row.unique():
        prev_sub_brands = df_merged[df_merged.row == i]['Sub-Brand_x'].values
        after_sub_brands = df_merged[df_merged.row == i]['Sub-Brand_y'].values

        match = list((Counter(prev_sub_brands) & Counter(after_sub_brands)).elements())
        if len(prev_sub_brands) == 0:
            row_level_compliance[i] = 0
        else:
            row_level_compliance[i] = 100 * len(match)/len(prev_sub_brands)
            
        row_level_prev_count[i] = len([j for j in prev_sub_brands if str(j).lower() != 'nan'])
        row_level_after_count[i] = len([j for j in after_sub_brands if str(j).lower() != 'nan'])
        
    
    # generating positional data
    final_dict = {}
    for i in df_merged.iterrows():
        i = i[1]
        row_id = i['row']
        position_id = i['column']
        
        if row_id not in final_dict.keys():
            final_dict[row_id] = {}
     
        row_id_dict = {}
        row_id_dict['sub_brand_prev'] = i['Sub-Brand_x']
        row_id_dict['sub_brand_after'] = i['Sub-Brand_y']
        row_id_dict['right_position'] = i['Compliance']
        final_dict[row_id][position_id] = row_id_dict
      
    for i in final_dict.keys():
        final_dict[i]['row_compliance'] = row_level_compliance[i]
        final_dict[i]['prev_rack_row_product_capacity'] = row_level_prev_count[i]
        final_dict[i]['after_rack_row_product_capacity'] = row_level_after_count[i]
        
    final_dict['row_position_compliance'] = 100 * df_merged.Compliance.mean()
    final_dict['rack_compliance'] = overall_rack_compliance
    
    final_dict['rack_product_capacity_prev'] = str(np.array(list(row_level_prev_count.values())).sum())
    final_dict['rack_product_capacity_after'] = str(np.array(list(row_level_after_count.values())).sum())
    
    return final_dict

def get_bbox_dict(packets_json, key):
    final_labels = []
    stock_out = 0
    total_bbox = 0
    for i in packets_json:
        for j in i:
            bbox = j.get_bbox()
            label = j.get_label()
            total_bbox += 1
            if label == 'Empty':
                stock_out += 1
            final_labels.append({'label':str(label), 'xmin':float(bbox[0]), 'ymin':float(bbox[1]), 
                                 'xmax':float(bbox[2]), 'ymax':float(bbox[3])})
    return {key : final_labels}, stock_out, 100*stock_out/total_bbox if total_bbox else 0
            
def match_prev_after_name(all_names, filter_name= None):
    '''
    all_names: list of all file names
    filter_name: if output is required only for one name
    '''

    prev_names = [i for i in all_names if ('prev' in i)]
    after_names = [i for i in all_names if ('after' in i)]
    
    prev_names_df = standardize_names(prev_names, 'prev')
    after_names_df = standardize_names(after_names, 'after')
    
    df_final = prev_names_df.merge(after_names_df, on= 'comparison_string')
    df_final.index = df_final['actual_name_x']
    df_final = df_final['actual_name_y'].to_dict()
    
    if filter_name:
        return df_final[filter_name]
    
    return df_final
                
      
class Compliance:
    
    def __init__(self, output_dict=None):
        '''
        output_dict: If passed as param, then class would not load the pickle files
            {'integrated_json_name': 'abc.json', output:same content that was saved in pickle}
        '''
        
        config = get_config("production_config")
        self.final_compliance_save = os.path.join(config['path']['blob_base_dir'], 
                                                  config['compliance']['path']['output_prev_after_comparsion'])
        
        try:
            os.mkdir(self.final_compliance_save)
        except:
            print('Directory already exists')
            
        if output_dict:
            self.output_dict = {key + '.pkl' : value for key, value in output_dict.items()}
            self.all_pickle_names = self.output_dict.keys()
            self.prev_after_mapping = match_prev_after_name(self.all_pickle_names)
            
        else:
            self.pickle_folder_path = os.path.join(config['path']['blob_base_dir'], config['compliance']['path']['output'])
            self.all_pickle_names = os.listdir(self.pickle_folder_path)
            self.all_pickle_names = [i for i in self.all_pickle_names if 'pkl' in i]
            self.prev_after_mapping = match_prev_after_name(self.all_pickle_names)
        
        
    
    
    def handle_missing_top_row(self, rack_rows_prev, rack_rows_after, packets_prev, image_height):
        '''adds an additional rack row and packets row if extra racks are present in the after image'''

        locator = Locator()
        cpu_config = get_config("production_config")
        thresholds = cpu_config['compliance']['threshold']

        avg_row_height_prev = locator.get_average_height(rack_rows_prev)
        additional_row, top_bottom = locator.find_missing_row_prev_after(rack_rows_prev=rack_rows_prev, 
                                                                         avg_height_prev=avg_row_height_prev, 
                                                                         rack_rows_after=rack_rows_after,
                                                                         image_height=image_height)

        if additional_row is None:
            return packets_prev

        if top_bottom == 'top':
            rack_rows_prev = np.insert(rack_rows_prev, 0, additional_row)
            nearest_row_idx = 0
        else:
            rack_rows_prev = np.insert(additional_row, 0, rack_rows_prev)
            nearest_row_idx = -1
              

        # average width of packets in the previous first row
        avg_width = locator.get_average_width(packets_prev[nearest_row_idx])

        packets = []
        for i in packets_prev:
            for j in i:
                packets.append(j)     
        packets = np.array(packets)

        if math.isnan(float(avg_width)):
            avg_width = locator.get_average_width(packets)

        if math.isnan(float(avg_width)):
            return packets_prev

        number_of_cols_width = (rack_rows_prev[nearest_row_idx].get_bbox()[2] - rack_rows_prev[nearest_row_idx].get_bbox()[0])/avg_width
        num_empty = locator.thresh_round(number_of_cols_width, thresh=thresholds['empty_row_bbox_thresh'])

        empty_space = {"empty_count": int(num_empty),
                                         "after": -1}

        y1 = rack_rows_prev[nearest_row_idx].get_bbox()[1]
        y2 = rack_rows_prev[nearest_row_idx].get_bbox()[3]

        start = rack_rows_prev[nearest_row_idx].get_bbox()[0]
        end = rack_rows_prev[nearest_row_idx].get_bbox()[2] 
        avg_width = (end - start)/empty_space['empty_count']

        if empty_space['empty_count'] == 0:
            return packets_prev
        
        for i in range(empty_space['empty_count']):
            empty_bbox = Packet("Empty", start + (avg_width*i), y1, end - (avg_width*(empty_space['empty_count']-1-i)), y2)
            if i == 0:
                row = np.array([empty_bbox])
            else:
                row = np.insert(row, empty_space['after']+1+i, empty_bbox)

        if top_bottom == 'top':
            packets_prev_final = [None]
            for i in packets_prev:
                packets_prev_final.append(i)
#             packets_prev = np.append(None, packets_prev)
            packets_prev = packets_prev_final
        else:
            packets_prev = list(packets_prev)
            packets_prev.append([None])
#             packets_prev = np.append(packets_prev, None)
            
        packets_prev[nearest_row_idx] = row

        return packets_prev

    def read_image_height(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
            
        height = data['image_dimension'][1]
        return height
    
    def read_image_path(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
            
        path = os.path.join(data['image_dir'], data['image_name'])
        return path
            
        
        
    def calculate_compliance(self, images_folder=None, model_output_folder=None):
            
        df_final = {'Image Name' : [], 'Detections' : []}
                                
        for prev, after in self.prev_after_mapping.items():
            
            if self.output_dict:
                output_prev = self.output_dict[prev]
                output_after = self.output_dict[after]
                
            else:   
                # pickle files have bouding boxes and positional information of all packets
                with open(os.path.join(self.pickle_folder_path, prev), 'rb') as file:
                    output_prev = pickle.load(file)
                
                with open(os.path.join(self.pickle_folder_path, after), 'rb') as file:
                    output_after = pickle.load(file)
                
            # loading the json file for detecting image dimensions
            integration_config = get_config("production_config")
            json_path_prev = os.path.join(integration_config['data_path']['blob_base_dir'], 
                                          integration_config['integrated_output']['integrated_dir'],
                                          integration_config['data_path']['output_images_folder'],
                                          prev.replace('.pkl', ''))
            
            json_path_after = os.path.join(integration_config['data_path']['blob_base_dir'], 
                                          integration_config['integrated_output']['integrated_dir'],
                                          integration_config['data_path']['output_images_folder'],
                                          after.replace('.pkl', ''))
            
            image_height_prev = self.read_image_height(json_path_prev)
            image_height_after = self.read_image_height(json_path_after)
            
            image_path_prev = self.read_image_path(json_path_prev)
            image_path_after = self.read_image_path(json_path_after)
                            
            # packets
            packets_prev = output_prev[0]
            packets_after = output_after[0]
            
            # rack rows
            rack_rows_prev = output_prev[1]
            rack_rows_after = output_after[1]
            
            
            if ((len(rack_rows_prev) == 0) & (len(packets_prev) == 0)) | ((len(rack_rows_after) == 0) & (len(packets_after) == 0)):
                pass
            
            else:
                # TODO: handle the case to go with one rack
                
                # handling cases where more racks in the after image
                if len(rack_rows_after) > len(rack_rows_prev):
                    packets_prev = self.handle_missing_top_row(rack_rows_prev=rack_rows_prev, rack_rows_after=rack_rows_after, 
                                                               packets_prev=packets_prev, image_height= image_height_prev)
                
                # handling cases where more racks in the prev image
                if len(rack_rows_prev) > len(rack_rows_after):
                    packets_after = self.handle_missing_top_row(rack_rows_prev=rack_rows_after, rack_rows_after=rack_rows_prev, 
                                                               packets_prev=packets_after, image_height= image_height_after)
                
            ordered_detections = get_image_compliance(packets_prev, packets_after)
            final_compliance = {}
            final_compliance['ordered_detections'] = ordered_detections
            
            
            rendering_detection, stock_out_prev, stock_out_perc_prev = get_bbox_dict(packets_json=packets_prev, key='packets_previous') 
            packets_after, stock_out_after, stock_out_perc_after = get_bbox_dict(packets_json=packets_after, key='packets_after')
            
            rendering_detection['packets_after'] = packets_after['packets_after']
            
            rendering_detection['image_path_prev'] = image_path_prev
            rendering_detection['image_path_after'] = image_path_after
            
            image_path_dict = {'path' : "/".join(image_path_prev.split("/")[0:-1]),
                               'prev_image_name' : image_path_prev.split("/")[-1],
                               'after_image_name' : image_path_after.split("/")[-1]}
            
            final_compliance['rendering_detection'] = rendering_detection
            final_compliance['stock_out'] = {'Previous Stock Out':stock_out_prev,
                            'Previous Stock Out Percentage':stock_out_perc_prev,
                            'After Stock Out':stock_out_after,
                            'After Stock Out Percentage':stock_out_perc_after}
            
            final_compliance['image_path'] = image_path_dict
            
            df_final['Detections'].append(final_compliance)
                        
            df_final['Image Name'].append(prev.replace(".json.pkl", ""))
            
            save_final_json(image_name=prev.replace(".json.pkl", ""),
                            json_result= final_compliance, json_path=self.final_compliance_save)

                
            if images_folder:
                image_prev = Image.open(images_folder + prev.replace('.pkl', '.png'))
                image_after = Image.open(images_folder + after.replace('.pkl', '.png'))
                
                fig = plt.figure(figsize=(50, 50))
                fig.add_subplot(1, 2, 1)
                ax = plt.imshow(image_prev)
                plt.axis("off")
                
                fig.add_subplot(1, 2, 2)
                ax = plt.imshow(image_after)
                plt.axis("off")
                
                plt.show();
                
                
                if model_output_folder:
                    json_result_prev = model_output_folder + prev.replace('.pkl', '')
                    image_prev = visualize_integrated_output(json_result_prev, visualization='all')
                    
                    json_result_after = model_output_folder + after.replace('.pkl', '')
                    image_after = visualize_integrated_output(json_result_after, visualization='all')
                    
                    fig = plt.figure(figsize=(50, 50))
                    fig.add_subplot(1, 2, 1)
                    ax = plt.imshow(image_prev)
                    plt.axis("off")
                    
                    fig.add_subplot(1, 2, 2)
                    ax = plt.imshow(image_after)
                    plt.axis("off")
                    
                    plt.show();
                    
        save_final_csv(pd.DataFrame(df_final), self.final_compliance_save)
    
    
    
class RenderFinalResults:
    def __init__(self, folder):
        self.folder = folder
        self.json_outputs=os.listdir(folder)
         
        
   
    def plot_image(self, image_path, packets, ax):
        image = plt.imread(image_path)
        im = np.array(image, dtype=np.uint8)
        ax.imshow(im)

        for i, packet in enumerate(packets):
            name = packet['label']
            left = int(packet['xmin'])
            bottom = int(packet['ymin'])
            right = int(packet['xmax'])
            top = int(packet['ymax'])
            width = right-left
            height = top - bottom

            if name == "Blank":
                rect = patches.Rectangle((left, bottom),width,height,linewidth=3,edgecolor='blue',facecolor='white', alpha=0.3)
            else:
                rect = patches.Rectangle((left, bottom),width,height,linewidth=3,edgecolor='blue',facecolor='white', alpha=0.15)
            ax.add_patch(rect)
            ax.text((left)+2, (bottom+top)/2, name, color= 'black', size= 10)
            
    def render_images(self):
        for i in self.json_outputs:
            with open(os.path.join(self.folder, i)) as f:
                data = json.load(f)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,30))
            self.plot_image(data['rendering_detection']['image_path_prev'], data['rendering_detection']['packets_previous'], ax1)
            self.plot_image(data['rendering_detection']['image_path_after'], data['rendering_detection']['packets_after'], ax2)  
            # print(plt)

            plt.savefig(os.path.join('/media/premium/common-biscuit/main/planogram_biscuit/data/raw/compliance_output',i.split(".")[0]+'.jpg'))
            print(i.split(".")[0]+'.jpg','Saved')