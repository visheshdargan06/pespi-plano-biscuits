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


def find_packet_positions(data):
    rack_rows, packets, complete_rack = locator.get_rows_packets_predictions(data)
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
