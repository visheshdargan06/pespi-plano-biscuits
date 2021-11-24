import os
import sys
import json
import pickle
import pathlib
import numpy as np
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from scoring.Evaluator import Evaluator
from utils.config import get_config
from utils.custom_datatypes import Packet, Row
from scoring.Locator import Locator


config = get_config("production_config")
#model_config = get_config("model_integration")

locator = Locator()
thresholds = config['compliance']['threshold']

def get_row_rack_overlap(rack, rack_rows, overlap_thresh=0.7):
    mask_list = []
    for row_num, row in enumerate(rack_rows):
        box_area = Evaluator._getArea([int(row['x1']), 
                                 int(row['y1']), 
                                 int(row['x2']), 
                                 int(row['y2'])])
        intersection_area = Evaluator._getIntersectionArea([int(rack['x1']), 
                                 int(rack['y1']), 
                                 int(rack['x2']), 
                                 int(rack['y2'])], [int(row['x1']), 
                                 int(row['y1']), 
                                 int(row['x2']), 
                                 int(row['y2'])])

        overlap_perc = intersection_area/box_area

        if overlap_perc > overlap_thresh:
            mask_list.append(True)
        else:
            mask_list.append(False)
        
    return np.array(mask_list)

def get_one_rack(output_dict, row_overlap_thresh=0.7):
    updated_dict = {}
    if len(output_dict['complete_rack'])>1:
        updated_dict['image_name'] = output_dict['image_name']
        max_conf = 0
        for i, conf in enumerate(output_dict['complete_rack_confidence']):
            if conf > max_conf:
                max_conf = conf
                max_index = i

        updated_dict['complete_rack'] = [output_dict['complete_rack'][max_index]]
        updated_dict['complete_rack_confidence'] = [max_conf]

        mask_list = get_row_rack_overlap(output_dict['complete_rack'][max_index], output_dict['rackrow'], row_overlap_thresh)
        print(mask_list)
        updated_dict['rackrow'] = (np.array(output_dict['rackrow'])[mask_list]).tolist()
        updated_dict['rackrow_confidence'] = np.array(output_dict['rackrow_confidence'])[mask_list].tolist()

        return updated_dict
        
    else:
        return output_dict

def find_packet_positions(data):
    # Handle two racks
    updated_data = get_one_rack(data)
    rack_rows, packets, complete_rack = locator.get_rows_packets_predictions(updated_data)
    rack_rows = locator.get_sorted_rows(rack_rows)
    avg_row_height = locator.get_average_height(rack_rows)
    
    rack_rows = locator.find_missing_row_between(rack_rows, avg_row_height, 
                                                 thresholds['missing_row_thresh'],
                                                 thresholds['missing_mid_row_thresh'])
    
    # first_row = locator.find_missing_first_row_modified(rack_rows, avg_row_height, packets, complete_rack,
    #                                            thresholds['missing_top_row_packet_overlap_thresh'], 
    #                                            thresholds['top_row_additional_pixels'], 
    #                                            thresholds['top_row_additional_thresh_pixels'])
    
    # if first_row is not None:
    #     rack_rows = np.insert(rack_rows, 0, first_row)

    # Check which packet is in which row.
    packet_positions = locator.check_packet_row(rack_rows, packets, 
                                                thresholds['packet_row_overlap_thresh'],
                                                thresholds['packet_top_row_overlap_thresh'])
    
    # Fill blank bboxes.
    updated_packet_positions = locator.fill_blank_bbox(packets, packet_positions, rack_rows,
                                                       thresholds['empty_row_bbox_thresh'],
                                                       thresholds['blank_middle_bbox_thresh'],
                                                       thresholds['blank_start_bbox_thresh'],
                                                       thresholds['blank_end_bbox_thresh'])
    
    #Find Blank boxes that are overlapping with actual boxes.
    overlap_blank = locator.find_overlapping_blank_bbox(updated_packet_positions, packets, 
                                                        iou_thresh=thresholds['blank_predicted_overlap_iou'])
                        
    # Remove blank bounding boxes that overlap other boxes.
    updated_packet_positions = locator.remove_overlapping_bbox(overlap_blank, updated_packet_positions)
    return updated_packet_positions, rack_rows

def locate_packets(pred_dir, predictions): 
    
    location_dict = {}
    for prediction in predictions:
        
        with open(os.path.join(pred_dir,prediction)) as file:
            data = json.loads(file.read())
        
        updated_packet_positions, rack_rows = find_packet_positions(data)
        location_dict[prediction] = [np.array(updated_packet_positions), np.array(rack_rows)]
 
    return location_dict

def locate_packets_pickle(pred_dir, predictions):
    output_dir = os.path.join(config['path']['blob_base_dir'],
                              config['compliance']['path']['output'])
    for prediction in predictions:

            with open(os.path.join(pred_dir,prediction)) as file:
                data = json.loads(file.read())

            updated_packet_positions, rack_rows = find_packet_positions(data)

            # Save pickle file
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_dir,
                               prediction+".pkl"), 'wb') as file:
                pickle.dump([np.array(updated_packet_positions), np.array(rack_rows)], file)

if __name__ == '__main__':
    
    pred_dir = os.path.join(model['data_path']['blob_base_dir'],
                             model['integrated_output']['integrated_dir'],
                             model['data_path']['output_images_folder'])
    output_dir = os.path.join(config['path']['blob_base_dir'],
                              config['compliance']['path']['output'])
    # Read image predictions
    #pred_dir = "/mnt/common/data/output/image_annotations/integrated/sample_1007"
    predictions = os.listdir(pred_dir)
    
    
    for prediction in predictions:
        
        with open(os.path.join(pred_dir,prediction)) as file:
            data = json.loads(file.read())
        
        updated_packet_positions, rack_rows = find_packet_positions(data)
        
        # Save pickle file 
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
        with open(os.path.join(output_dir,
                               prediction+".pkl"), 'wb') as file:
            pickle.dump([np.array(updated_packet_positions), np.array(rack_rows)], file)    
