import sys
sys.path.append("../")

import numpy as np
import math
from utils.bb_utils import get_width, get_height
from scoring.Evaluator import Evaluator
from utils.custom_datatypes import Packet, Row, CompleteRack


class Locator: 
    
    
    def __init__(self):
        pass
    
    def get_sorted_rows(self, rack_rows):

        # Sort the rows
        return np.array(sorted(rack_rows, key=lambda x: x.get_bbox()[1]))
    
    def get_average_height(self, boxes):
        return np.mean(np.array([get_height(box.get_bbox()) for box in boxes[1:]]))
    
    def get_average_width(self, boxes):
        return np.mean(np.array([get_width(box.get_bbox()) for box in boxes]))
    
    def packet_overlap(self, rack_rows, packet, overlap_thresh=0.7, top_overlap_thresh=0.7):
    
        box_area = Evaluator._getArea(packet.get_bbox())
        for row_num, row in enumerate(rack_rows):
            intersection_area = Evaluator._getIntersectionArea(row.get_bbox(), packet.get_bbox())
            overlap_perc = intersection_area/box_area
            # Currently returns only the first row it overlaps with
            if row_num==0:
                if overlap_perc > top_overlap_thresh:
                    return row_num
            else:
                if overlap_perc > overlap_thresh:
                    return row_num
        return -1
    
    def get_sorted_columns(self, packet_positions):
    
        for row_num, row in enumerate(packet_positions):
            # Sort packets by column
            packet_positions[row_num] = np.array(sorted(row, key=lambda x: x.get_bbox()[0]))

        return packet_positions
    
    def thresh_round(self, number, thresh=0.5):
        if number < 0:
            return 0
        if number > thresh:
            return np.ceil(number)
        return np.floor(number)
    
    def thresh_round_row(self, number, thresh=0.5, fixing_number_mid_row = 2.2):
        if number < 0:
            return 0
        if number > thresh:
            return np.ceil(number/fixing_number_mid_row)
        return np.floor(number)
    
    def get_rows_packets_predictions(self, data):
    
        rack_rows = []
        for row in data['row_boxes']:
            rack_rows.append(Row(int(row['x1']), 
                                 int(row['y1']), 
                                 int(row['x2']), 
                                 int(row['y2'])))

        packets = []
        for packet in data['packets']:
            packets.append(Packet(packet['sub_brand'], 
                                  int(packet['x1']), 
                                  int(packet['y1']), 
                                  int(packet['x2']), 
                                  int(packet['y2'])))

        complete_rack = []

        for rack in data['complete_rack']:
            complete_rack.append(CompleteRack(int(rack['x1']), 
                                  int(rack['y1']), 
                                  int(rack['x2']), 
                                  int(rack['y2'])))

        packets = np.array(packets)
        rack_rows = np.array(rack_rows)
        complete_rack = np.array(complete_rack)

        return rack_rows, packets,complete_rack
    
    def get_rows_packets_annotations(root_element):
        rack_rows = []
        packets = []
        for obj in root_element.findall('object'):
            name = obj.find('name').text

            for boxes in obj.findall('bndbox'):

                x1 = int(boxes.find('xmin').text)
                y1 = int(boxes.find('ymin').text)
                x2 = int(boxes.find('xmax').text)
                y2 = int(boxes.find('ymax').text)
                #box = np.array([x1, y1, x2, y2])

                if name == "Rack Row":
                    rack_rows.append(Row(x1, y1, x2, y2))
                    #rack_rows.append(box)
                elif name != "Complete Rack":
                    packets.append(Packet(name, x1, y1, x2, y2))
                    #packets.append(box)
        packets = np.array(packets)
        rack_rows = np.array(rack_rows)

        return rack_rows, packets
    
    def find_missing_row_between(self, rack_rows, avg_row_height, thresh=0.8 , fixing_number_mid_row = 2.2):   
        # Find missed rows in-between
        updated_rack_rows = []
        for row_num, row in enumerate(rack_rows):
            updated_rack_rows.append(row)

            if row_num+1 == len(rack_rows):
                break

            missing_rows = (rack_rows[row_num+1].get_bbox()[1] - row.get_bbox()[3])/avg_row_height
            missing_rows = self.thresh_round_row(missing_rows, thresh=thresh, fixing_number_mid_row = fixing_number_mid_row )

            if missing_rows > 0:
                empty_space = {'empty_count': int(missing_rows),
                               'after': row_num}
                rows = self.fill_empty_row(row.get_bbox()[3], rack_rows[row_num+1].get_bbox()[1], rack_rows, empty_space)
                updated_rack_rows.extend(rows)

        rack_rows = np.array(updated_rack_rows)

        return self.get_sorted_rows(rack_rows)
    
    def find_missing_first_row(self, rack_rows, avg_height, packets, overlap_thresh=0.7, add_pixels=25, thresh_pixels=50):
        try:
            x1 = np.max(np.array([row.get_bbox()[0] for row in rack_rows])) 
        except Exception as e:
            print("No rack rows present")
            return None
        y1 = rack_rows[0].get_bbox()[1] - avg_height
        if y1 > thresh_pixels:
            y1 = y1 - add_pixels
        x2 = np.max(np.array([row.get_bbox()[2] for row in rack_rows]))
        y2 = rack_rows[0].get_bbox()[1]
        row = Row(x1, y1, x2, y2)
        # check if y1 coordinate becoame -ve.
        if y1 < 0:
            return None

        # check if any packets in this row.
        for packet in packets:
            box_area = Evaluator._getArea(packet.get_bbox())
            intersection_area = Evaluator._getIntersectionArea(row.get_bbox(), packet.get_bbox())
            overlap_perc = intersection_area/box_area

            if overlap_perc > overlap_thresh:
                return row
        return None

    
    def find_missing_first_row_modified(self, rack_rows, avg_row_height, packets, complete_rack, overlap_thresh=0.7, add_pixels=25, thresh_pixels=50):
        if abs(complete_rack[0].get_bbox()[1] - rack_rows[0].get_bbox()[1]) > avg_row_height:
            try:
                x1 = np.max(np.array([row.get_bbox()[0] for row in rack_rows])) 
            except Exception as e:  
                print("No rack rows present")
                return None
            y1 = rack_rows[0].get_bbox()[1] - avg_row_height
            if y1 > thresh_pixels:
                y1 = y1 - add_pixels
            x2 = np.max(np.array([row.get_bbox()[2] for row in rack_rows]))
            y2 = rack_rows[0].get_bbox()[1]
            row = Row(x1, y1, x2, y2)
            # check if y1 coordinate becoame -ve.
            if y1 < 0:
                return None
            
            # check if any packets in this row.
            for packet in packets:
                box_area = Evaluator._getArea(packet.get_bbox())
                intersection_area = Evaluator._getIntersectionArea(row.get_bbox(), packet.get_bbox())
                overlap_perc = intersection_area/box_area

                if overlap_perc > overlap_thresh:
                    return row
            return None
        else:
            return None
    
    def find_missing_row_prev_after(self, rack_rows_prev, avg_height_prev, rack_rows_after, image_height,
                                    overlap_thresh=0.7, add_pixels=25, thresh_pixels=50, top_row_compression_thres=0.7):

        if not len(rack_rows_after) > len(rack_rows_prev):
            return None, None

        avg_height_prev = avg_height_prev * top_row_compression_thres

        # case 1 : top row
        x1_top = np.max(np.array([row.get_bbox()[0] for row in rack_rows_prev]))
        y1_top = rack_rows_prev[0].get_bbox()[1] - avg_height_prev


        x2_top = np.max(np.array([row.get_bbox()[2] for row in rack_rows_prev]))
        y2_top = rack_rows_prev[0].get_bbox()[1]

        # case 2 : bottom row
        x1_bottom = np.max(np.array([row.get_bbox()[0] for row in rack_rows_prev]))
        y1_bottom = rack_rows_prev[-1].get_bbox()[3]

        x2_bottom = np.min(np.array([row.get_bbox()[2] for row in rack_rows_prev]))
        y2_bottom = rack_rows_prev[-1].get_bbox()[3] + avg_height_prev

        
        if y1_top > thresh_pixels:
            y1_top = y1_top - add_pixels

        
        if y1_top > 0: # top row can be accomodated
            row = Row(x1_top, y1_top, x2_top, y2_top)
            return row, 'top'
        elif y2_bottom < image_height: # bottom row can be accomodated
            row = Row(x1_bottom, y1_bottom, x2_bottom, y2_bottom)
            return row, 'bottom'
        else:
            return None, None

        
        
    
    def fill_empty_row(self, start, end, rack_rows, empty_space):
    
        x1 = np.median(np.array([row.get_bbox()[0] for row in rack_rows]))
        x2 = np.median(np.array([row.get_bbox()[2] for row in rack_rows]))
        avg_height = (end - start)/empty_space['empty_count']
        rows = []
        for i in range(empty_space['empty_count']):
            row_bbox = Row(x1, start + (avg_height*i), x2, end - (avg_height*(empty_space['empty_count']-1-i)))
            rows.append(row_bbox)

        return rows
    
    def fill_empty_bbox_empty_row(self, start, end, row, empty_space, avg_width, rack):
    
        y1 = rack.get_bbox()[1]
        y2 = rack.get_bbox()[3]
        avg_width = (end - start)/empty_space['empty_count']
        for i in range(empty_space['empty_count']):
            empty_bbox = Packet("Empty", start + (avg_width*i), y1, end - (avg_width*(empty_space['empty_count']-1-i)), y2)
            if row.shape[0] == 0:
                row = np.array([empty_bbox])
            else:
                row = np.insert(row, empty_space['after']+1+i, empty_bbox)

        return row
    
    def fill_empty_bbox(self, start, end, row, empty_space, avg_width):
    
        y1 = np.mean(np.array([packet.get_bbox()[1] for packet in row]))
        y2 = np.mean(np.array([packet.get_bbox()[3] for packet in row]))
        avg_width = (end - start)/empty_space['empty_count']
        for i in range(empty_space['empty_count']):
            empty_bbox = Packet("Empty", start + (avg_width*i), y1, end - (avg_width*(empty_space['empty_count']-1-i)), y2)
            row = np.insert(row, empty_space['after']+1+i, empty_bbox)

        return row
    
    def check_packet_row(self, rack_rows, packets, overlap_thresh=0.7, top_overlap_thresh=0.7):
        
        packet_positions = np.empty((rack_rows.shape[0],), dtype=object)
    
        for i,v in enumerate(packet_positions): packet_positions[i]=[]

        for packet in packets:
            row_num = self.packet_overlap(rack_rows, packet, overlap_thresh, top_overlap_thresh)
            if row_num != -1:
                packet_positions[row_num].append(packet)

        packet_positions = self.get_sorted_columns(packet_positions)
        
        return packet_positions
    
    def fill_blank_bbox(self, packets, packet_positions, rack_rows,
                        empty_row_bbox_thresh=0.5,
                        blank_middle_bbox_thresh=0.5,
                        blank_start_bbox_thresh=0.5,
                        blank_end_bbox_thresh=0.5):
        
        updated_packet_positions = []
        for row_num, row in enumerate(packet_positions):
            updated_row = row
            avg_width = self.get_average_width(row)

            ## Use this only when row is empty/no packet detected.
            if len(row) == 0:
                updated_row = self._fill_blank_bbox_empty_row(packets, row_num, 
                                                              packet_positions, 
                                                              rack_rows,
                                                              updated_row,
                                                              empty_row_bbox_thresh)
                updated_packet_positions.append(np.array(updated_row))
                continue
            
            updated_row = self._fill_blank_bbox_middle(row, rack_rows, 
                                                       avg_width, updated_row,
                                                       blank_middle_bbox_thresh)

            # After middle box fill because middle box fill chexks number of bboxx identified by the model.
            updated_row = self._fill_blank_bbox_start(row, rack_rows, 
                                                      row_num, avg_width, 
                                                      updated_row,
                                                      blank_start_bbox_thresh)
            updated_row = self._fill_blank_bbox_end(row, rack_rows, 
                                                    row_num, avg_width, 
                                                    updated_row,
                                                    blank_end_bbox_thresh)

            updated_packet_positions.append(np.array(updated_row))
            
        return updated_packet_positions
    
    def _fill_blank_bbox_empty_row(self, packets, row_num, packet_positions, rack_rows, updated_row, thresh=0.5):
        
        try:
            if row_num-1 < 0:
                avg_width = self.get_average_width(packet_positions[row_num+1])
            else:
                avg_width = self.get_average_width(packet_positions[row_num-1])
        except:
            print("Single Row")
            return updated_row

        if math.isnan(float(avg_width)):
            avg_width = self.get_average_width(packets)

        if math.isnan(float(avg_width)):
            empty_flag = 1
            # No packet in rack
            return updated_row

        number_of_cols_width = (rack_rows[row_num].get_bbox()[2] - rack_rows[row_num].get_bbox()[0])/avg_width

        num_empty = self.thresh_round(number_of_cols_width, thresh=thresh)

        ## Fill in empty spaces.
        empty_space = {"empty_count": int(num_empty),
                                     "after": -1}
        updated_row = self.fill_empty_bbox_empty_row(rack_rows[row_num].get_bbox()[0], rack_rows[row_num].get_bbox()[2], updated_row, empty_space, avg_width, rack_rows[row_num])
        
        return updated_row
    
    def _fill_blank_bbox_middle(self, row, rack_rows, avg_width, updated_row, thresh=0.5):
        
        number_of_cols_width_bbox = (row[-1].get_bbox()[2] - rack_rows[0].get_bbox()[0])/avg_width
        
        # Find blank bbox only if number of identified boxes is less than total bbox it can fit.
        if len(row) < self.thresh_round(number_of_cols_width_bbox, thresh=thresh):

            for packet_num, packet in enumerate(row):

                if packet_num+1 == len(row):
                    break

                number_of_cols_width_packet = (row[packet_num+1].get_bbox()[0] - packet.get_bbox()[2])/avg_width
                num_spaces = self.thresh_round(number_of_cols_width_packet, thresh=thresh)
                if num_spaces > 0:
                    empty_space = {"empty_count": int(num_spaces),
                                         "after": packet_num}
                    updated_row = self.fill_empty_bbox(packet.get_bbox()[2], row[packet_num+1].get_bbox()[0], updated_row, empty_space, avg_width)
        
        return updated_row
        
    def _fill_blank_bbox_start(self, row, rack_rows, row_num, avg_width, updated_row, thresh=0.5):
        
        num_start_empty = self.thresh_round((row[0].get_bbox()[0] - rack_rows[row_num].get_bbox()[0])/avg_width, thresh=thresh)
        if num_start_empty > 0:
            empty_space = {"empty_count": int(num_start_empty),
                                         "after": -1}
            updated_row = self.fill_empty_bbox(rack_rows[row_num].get_bbox()[0], row[0].get_bbox()[0], updated_row, empty_space, avg_width)
            
        return updated_row
    
    def _fill_blank_bbox_end(self, row, rack_rows, row_num, avg_width, updated_row, thresh=0.5):
        
        num_end_empty = self.thresh_round((rack_rows[row_num].get_bbox()[2] - row[-1].get_bbox()[2])/avg_width, thresh=thresh)
        if num_end_empty > 0:
            empty_space = {"empty_count": int(num_end_empty),
                                         "after": len(updated_row)-1}
            updated_row = self.fill_empty_bbox(updated_row[-1].get_bbox()[2], rack_rows[row_num].get_bbox()[2], updated_row, empty_space, avg_width)
            
        return updated_row
    
    def find_overlapping_blank_bbox(self, updated_packet_positions, packets, iou_thresh = 0.5):
        
        overlap_blank = {}
        for i, row in enumerate(np.array(updated_packet_positions)):
            overlap_blank[i] = []
            for j, packet in enumerate(row):

                if packet.get_label() == "Empty":
                    for actual_packet in packets:
                        iou = Evaluator.iou(actual_packet.get_bbox(), packet.get_bbox())

                        if iou > iou_thresh:
                            overlap_blank[i].append(j)
                            break
                            
        return overlap_blank
    
    def remove_overlapping_bbox(self, overlap_blank, updated_packet_positions):
        
        if len(overlap_blank)==0:
            return updated_packet_positions
        
        for row, cols in overlap_blank.items():
            updated_packet_positions[row] = np.delete(updated_packet_positions[row], cols)
        
        return updated_packet_positions
