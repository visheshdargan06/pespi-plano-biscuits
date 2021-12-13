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

import subprocess as sbp

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

def get_rack_position(rack_coord, all_racks):
    for index,i in enumerate(all_racks):
        if i == rack_coord[0]:
            continue
        elif int(rack_coord[0]['x1']) < int(i['x1']):
            return "RIGHT"
        else:
            return "LEFT"

def select_one_rack_prev(rack_dict, complete_rack_confidence, rack_position):
    if len(rack_dict)>1:
        x1_1 = rack_dict[0]['x1']
        x1_2 = rack_dict[1]['x1']
        
        if rack_position == 'LEFT':
            if x1_1 < x1_2:
                return rack_dict[1], complete_rack_confidence[1]
            else:
                return rack_dict[0], complete_rack_confidence[0]
        elif rack_position == 'RIGHT':
            if x1_1 < x1_2:
                return rack_dict[0], complete_rack_confidence[0]
            else:
                return rack_dict[1], complete_rack_confidence[1]
        else:
            return None
    else:
        return None

def get_rack_position_from_after(prediction, position_dict_after):
    try:
        image_name = prediction.replace('prev','after')
        return position_dict_after[image_name]
    except:
        return None

def get_one_rack(output_dict, row_overlap_thresh=0.7):
    updated_dict = {}
    if len(output_dict['complete_rack'])>1:
        updated_dict['image_name'] = output_dict['image_name']
        updated_dict['image_dimension'] = output_dict['image_dimension']
        updated_dict['packets'] = output_dict['packets']
        updated_dict['packets_confidence'] = output_dict['packets_confidence']


        max_conf = 0
        for i, conf in enumerate(output_dict['complete_rack_confidence']):
            if conf > max_conf:
                max_conf = conf
                max_index = i

        updated_dict['complete_rack'] = [output_dict['complete_rack'][max_index]]
        updated_dict['complete_rack_confidence'] = [max_conf]

        updated_dict['rack_position_flag'] = get_rack_position(updated_dict['complete_rack'], output_dict['complete_rack'])

        mask_list = get_row_rack_overlap(output_dict['complete_rack'][max_index], output_dict['row_boxes'], row_overlap_thresh)
        #print(mask_list)
        updated_dict['row_boxes'] = (np.array(output_dict['row_boxes'])[mask_list]).tolist()
        updated_dict['row_boxes_confidence'] = np.array(output_dict['row_boxes_confidence'])[mask_list].tolist()
        
        updated_dict['image_dir'] = output_dict['image_dir']
        return updated_dict
        
    else:
        output_dict['rack_position_flag'] = None
        return output_dict

def get_one_rack_prev(output_dict, rack_position_prev, row_overlap_thresh=0.7):
    file_location = '/media/premium/common-biscuit/main/planogram_biscuit/data/output/compliance_visualization/compliance_output_updated'
    save_location = '/media/premium/common-biscuit/main/planogram_biscuit/data/output/validation/mismatch_racks_cases_compliance/'
    save_location_no_rack = '/media/premium/common-biscuit/main/planogram_biscuit/data/output/validation/no_rack_detected/'
    save_location_more_rack = '/media/premium/common-biscuit/main/planogram_biscuit/data/output/validation/more_than_2_racks/'

    updated_dict = {}
    #print(output_dict['image_name'], rack_position_prev, len(output_dict['complete_rack']))

    if len(output_dict['complete_rack']) == 0 and (rack_position_prev is None):
        sbp.call(['cp', os.path.join(file_location, output_dict['image_name'].split(".")[0]+'.jpg'), os.path.join(save_location_no_rack,(output_dict['image_name'].split(".")[0]+'.jpg'))])
        
    elif len(output_dict['complete_rack']) == 1 and (rack_position_prev is not None):
        sbp.call(['cp', os.path.join(file_location, output_dict['image_name'].split(".")[0]+'.jpg'), os.path.join(save_location,(output_dict['image_name'].split(".")[0]+'.jpg'))])
        
    elif len(output_dict['complete_rack'])>2:
        sbp.call(['cp', os.path.join(file_location, output_dict['image_name'].split(".")[0]+'.jpg'), os.path.join(save_location_more_rack,(output_dict['image_name'].split(".")[0]+'.jpg'))])
        

    if len(output_dict['complete_rack'])>1:
        if rack_position_prev is None:
            
            sbp.call(['cp', os.path.join(file_location, output_dict['image_name'].split(".")[0]+'.jpg'), os.path.join(save_location,(output_dict['image_name'].split(".")[0]+'.jpg'))])
            
            updated_data = get_one_rack(output_dict)
            return updated_data
        else:
            updated_dict['image_name'] = output_dict['image_name']
            updated_dict['image_dimension'] = output_dict['image_dimension']
            updated_dict['packets'] = output_dict['packets']
            updated_dict['packets_confidence'] = output_dict['packets_confidence']

            #print(updated_dict['image_name'], rack_position_prev, output_dict['complete_rack'])
            complete_rack, complete_rack_confidence = select_one_rack_prev(output_dict['complete_rack'], output_dict['complete_rack_confidence'], rack_position_prev)

            updated_dict['complete_rack'] = [complete_rack]
            updated_dict['complete_rack_confidence'] = [complete_rack_confidence]

            updated_dict['rack_position_flag'] = rack_position_prev

            mask_list = get_row_rack_overlap(updated_dict['complete_rack'][0], output_dict['row_boxes'], row_overlap_thresh)
            #print(mask_list)
            updated_dict['row_boxes'] = (np.array(output_dict['row_boxes'])[mask_list]).tolist()
            updated_dict['row_boxes_confidence'] = np.array(output_dict['row_boxes_confidence'])[mask_list].tolist()
            
            updated_dict['image_dir'] = output_dict['image_dir']
            return updated_dict
        
    else:
        output_dict['rack_position_flag'] = None
        return output_dict

def find_packet_positions(data):
    
    updated_data = get_one_rack(data)  # Handle two racks
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
    return updated_packet_positions, rack_rows, updated_data['rack_position_flag']

def find_packet_positions_prev(data, rack_position):
    updated_data = get_one_rack_prev(data, rack_position)  # Handle two racks
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
    return updated_packet_positions, rack_rows, updated_data['rack_position_flag']

def locate_packets(pred_dir, predictions): 
    
    location_dict = {}

    rack_position_dict = {}
    after_images_prediction = [pred for pred in predictions if "after" in pred]
    prev_images_prediction = [pred for pred in predictions if "prev" in pred]

    if len(after_images_prediction) and len(prev_images_prediction) > 1:
        for prediction in after_images_prediction:
            
            with open(os.path.join(pred_dir,prediction)) as file:
                data = json.loads(file.read())
            
            updated_packet_positions, rack_rows, rack_position_after = find_packet_positions(data)
            location_dict[prediction] = [np.array(updated_packet_positions), np.array(rack_rows)]
            rack_position_dict[prediction.replace('.json', '')] = rack_position_after

        for prediction in tqdm(prev_images_prediction):
            with open(os.path.join(pred_dir,prediction)) as file:
                data = json.loads(file.read())
            

            complete_rack_position = get_rack_position_from_after(prediction.replace('.json', ''), rack_position_dict)
            
            #print(rack_position_prev, len(data['complete_rack']))
            updated_packet_positions, rack_rows, rack_position = find_packet_positions_prev(data, complete_rack_position)
            location_dict[prediction] = [np.array(updated_packet_positions), np.array(rack_rows)]
    else: 
        for prediction in predictions:
            with open(os.path.join(pred_dir,prediction)) as file:
                data = json.loads(file.read())
            
            updated_packet_positions, rack_rows = find_packet_positions(data)
            location_dict[prediction] = [np.array(updated_packet_positions), np.array(rack_rows)]
 
    return location_dict

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
        
        updated_packet_positions, rack_rows, rack_position = find_packet_positions(data)
        
        # Save pickle file 
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
        with open(os.path.join(output_dir,
                               prediction+".pkl"), 'wb') as file:
            pickle.dump([np.array(updated_packet_positions), np.array(rack_rows)], file)    
