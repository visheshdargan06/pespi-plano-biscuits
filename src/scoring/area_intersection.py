def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        #update to exclude extra packets detected out of rack_row
        if xA>xB or yA>yB:
            return 0
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

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