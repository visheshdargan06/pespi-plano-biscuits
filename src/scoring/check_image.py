
import os
import sys
import json
import statistics
from utils.config import get_config
from scoring.Evaluator import Evaluator

config = get_config("production_config")
detection_threshold = config['image_check']['detection_threshold']
rack_area_threshold = config['image_check']['rack_area_threshold']
packets_area_threshold = config['image_check']['packets_area_threshold']
# rack_height_threshold = config['image_check']['rack_height_threshold']
# row_height_threshold = config['image_check']['row_height_threshold']
# packets_height_threshold = config['image_check']['packets_height_threshold']
# rack_width_threshold = config['image_check']['rack_width_threshold']
# row_width_threshold = config['image_check']['row_width_threshold']
# packets_width_threshold = config['image_check']['packets_width_threshold']
packet_count_threshold = config['image_check']['packet_count_threshold']
row_count_threshold = config['image_check']['row_count_threshold']

def get_avg_detection_size(list_data):
    height = []
    width = []

    for coordinates in list_data:
        height.append(int(coordinates['y2']) - int(coordinates['y1']))
        width.append(int(coordinates['x2']) - int(coordinates['x1']))

    avg_height = sum(height) / len(height)
    avg_width = sum(width) / len(width)
    return avg_height, avg_width    

def get_avg_area(data_list):
    areas = []
    for item in data_list:
        areas.append(Evaluator._getArea([int(item['x1']),int(item['y1']),int(item['x2']),int(item['y2'])]))
    return sum(areas) / len(areas)

def sanity_check(data):
    if len(data['complete_rack']) == 0:
        print("No rack found, compliance can't be calculated")
        return None

    elif len(data['complete_rack']) == 1:
        avg_rackrow_confidence = sum(data['rackrow_confidence']) / len(data['rackrow_confidence'])
        avg_packets_confidence = sum(data['packets_confidence']) / len(data['packets_confidence'])
        total_detection_confidence = ((0.2*avg_rackrow_confidence) + (0.8*avg_packets_confidence) + (0.1*data['complete_rack_confidence'])) #/3

        # avg_rackrow_height, avg_rackrow_width = get_avg_detection_size(data['rackrow'])
        # avg_packets_height, avg_packets_width = get_avg_detection_size(data['packets'])
        avg_rackrow_area = get_avg_area(data['row_boxes'])
        avg_packets_area = get_avg_area(data['packets'])

        row_count = len(data['rackrow'])
        packets_count = len(data['packets'])

        # if total_detection_confidence > detection_threshold and \
        #        avg_rackrow_height > rack_height_threshold and \
        #            avg_rackrow_width > row_width_threshold and \
        #                avg_packets_height > packets_height_threshold and \
        #                    avg_packets_width > packets_width_threshold and \
        #                        row_count > row_count_threshold and \
        #                            packets_count > packet_count_threshold:
        #     return True

        if total_detection_confidence > detection_threshold and \
            avg_rackrow_area > rack_area_threshold and \
                avg_packets_area > packets_area_threshold and \
                        row_count > row_count_threshold and \
                            packets_count > packet_count_threshold:
            return True

    elif len(data['complete_rack']) > 1:
        pass