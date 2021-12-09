
import os
import json
from utils.config import get_config
from scoring.Evaluator import Evaluator
from utils.utils import save_remove_images_json

config = get_config("production_config")
total_detection_threshold = config['image_check']['total_detection_threshold']
rack_area_threshold = config['image_check']['rack_area_threshold']
packets_area_threshold = config['image_check']['packets_area_threshold']
# rack_height_threshold = config['image_check']['rack_height_threshold']
# row_height_threshold = config['image_check']['row_height_threshold']
# packets_height_threshold = config['image_check']['packets_height_threshold']
# rack_width_threshold = config['image_check']['rack_width_threshold']
# row_width_threshold = config['image_check']['row_width_threshold']
# packets_width_threshold = config['image_check']['packets_width_threshold']
# packet_count_threshold = config['image_check']['packet_count_threshold']
row_count_threshold = config['image_check']['row_count_threshold']

remove_images_json_path = config['path']['remove_images_json']

def get_avg_detection_size(list_data):
    height = []
    width = []
    for coordinates in list_data:
        height.append(int(coordinates['y2']) - int(coordinates['y1']))
        width.append(int(coordinates['x2']) - int(coordinates['x1']))

    avg_height = sum(height) / len(height)
    avg_width = sum(width) / len(width)
    return avg_height, avg_width    

def get_avg_area(data_list, type=None):
    try:
        areas = []
        for item in data_list:
            if type == 'packets':
                areas.append((Evaluator._getArea([int(item['x1']),int(item['y1']),int(item['x2']),int(item['y2'])]))/1000)
            if type == 'rackrow':
                areas.append((Evaluator._getArea([int(item['x1']),int(item['y1']),int(item['x2']),int(item['y2'])]))/10000)
        return sum(areas) / len(areas)
    except:
        return 0

def sanity_check(data):
    if len(data['row_boxes']) == 0:
        return False, "No rows"

    elif len(data['row_boxes']) == 0 and len(data['complete_rack']) == 0:
        return False, "No rack and rows"

    else:
        avg_rackrow_confidence = sum(data['row_boxes_confidence']) / len(data['row_boxes_confidence'])
        avg_packets_confidence = sum(data['packets_confidence']) / len(data['packets_confidence']) if len(data['packets_confidence'])>0 else 0
        avg_rack_confidence = sum(data['complete_rack_confidence']) / len(data['complete_rack_confidence']) if len(data['complete_rack_confidence'])>0 else 0
        total_detection_confidence = ((0.2*avg_rackrow_confidence) + (0.8*avg_packets_confidence) + (0.1*avg_rack_confidence)) #/3

        # avg_rackrow_height, avg_rackrow_width = get_avg_detection_size(data['rackrow'])
        # avg_packets_height, avg_packets_width = get_avg_detection_size(data['packets'])
        avg_rackrow_area = get_avg_area(data['row_boxes'], type='rackrow')
        avg_packets_area = get_avg_area(data['packets'], type='packets')

        row_count = len(data['row_boxes'])
        #packets_count = len(data['packets'])

        if total_detection_confidence > total_detection_threshold and \
            avg_rackrow_area > rack_area_threshold and \
                avg_packets_area > packets_area_threshold and \
                        row_count > row_count_threshold :
            return True, "Pass"
        else:
            return False, "Thresholds not passed"

def image_sanity_check():
    packets_output_dir = config['path']['blob_base_dir'] + config['integrated_output']['packets_output_dir'] + config['data_path']['output_images_folder']
    rackrow_output_dir = config['path']['blob_base_dir'] + config['integrated_output']['rackrow_output_dir'] + config['data_path']['output_images_folder']
    # packets_output_dir = "/media/premium/common-biscuit/main/planogram_biscuit/data/output/image_annotations/packets_detection/op_annotations_new/"
    # rackrow_output_dir = "/media/premium/common-biscuit/main/planogram_biscuit/data/output/image_annotations/rackrow_detection/op_annotations_new/"

    remove_images = {}
    for json_output in os.listdir(rackrow_output_dir):
        if 'ipynb' in json_output:
            continue
        
        with open(packets_output_dir + json_output, 'r') as json_file:
            packets_output = json.load(json_file)
            
        with open(rackrow_output_dir + json_output, 'r') as json_file:
            rack_row_output = json.load(json_file)
    
        packets_output['row_boxes'] = rack_row_output['row_boxes']
        packets_output['row_boxes_confidence'] = rack_row_output['row_boxes_confidence']

        packets_output['complete_rack'] = rack_row_output['complete_rack']
        packets_output['complete_rack_confidence'] = rack_row_output['complete_rack_confidence']

        if packets_output['image_name'] not in remove_images:
            image_check, reason = sanity_check(packets_output)
            if image_check is False:
                remove_images[packets_output['image_name']] = reason
                if 'after' in packets_output['image_name']:
                    remove_images[packets_output['image_name'].replace('after', 'prev')] = "After image has issues"
                elif 'prev' in packets_output['image_name']:
                    remove_images[packets_output['image_name'].replace('prev', 'after')] = "Previous image has issues"

    save_remove_images_json(remove_images, remove_images_json_path)
    return remove_images