import sys
sys.path.append("../")

import os
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
from utils.bb_utils import *
from scoring.BoundingBox import BoundingBox
from scoring.BoundingBoxes import BoundingBoxes
import scoring.Evaluator

 
class MultiEvaluation:
    
    def __init__(self):
        
        pass
    
    def get_image_metrics(self, gt_boxes, pred_boxes, iou_thr, image):
        """
        Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

        Parameters:
            gt_boxes (list of list of floats): list of locations of ground truth
                objects as [{"class": "Cheetos",
                              "xmin": 10, 
                              "ymin": 20, 
                              "xmax": 30, 
                              "ymax": 40},
                            {"class": "Fritos",
                              "xmin": 102, 
                              "ymin": 203, 
                              "xmax": 340, 
                              "ymax": 405}]
            pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
                and 'scores' 
                [{"class": "Cheetos",
                          "xmin": 10, 
                          "ymin": 20, 
                          "xmax": 30, 
                          "ymax: 40,
                          "score": 0.12},
                        {"class": "Fritos",
                          "xmin": 102, 
                          "ymin": 203, 
                          "xmax": 340, 
                          "ymax": 405,
                          "score": 0.88}]
                
            iou_thr (float): value of IoU to consider as threshold for a
                true prediction.

        Returns:
            dict: true positives (int), false positives (int), false negatives (int)
        """
        
        image_bboxes = BoundingBoxes()
        gt_bbox=[]
        for bbox in gt_boxes:
            box = BoundingBox(imageName=image, classId=bbox['class'], 
                                          x=bbox['xmin'], y=bbox['ymin'], 
                                          w=bbox['xmax'], h=bbox['ymax'], 
                                          typeCoordinates=CoordinatesType.Absolute,
                                          bbType=BBType.GroundTruth, 
                                          format=BBFormat.XYX2Y2)
            gt_bbox.append(box)
            image_bboxes.addBoundingBox(box)
            
        pred_bbox=[]
        for bbox in pred_boxes:
            box = BoundingBox(imageName=image, classId=bbox['class'], 
                                          classConfidence=bbox['score'],
                                          x=bbox['xmin'], y=bbox['ymin'], 
                                          w=bbox['xmax'], h=bbox['ymax'],  
                                          typeCoordinates=CoordinatesType.Absolute,
                                          bbType=BBType.Detected, 
                                          format=BBFormat.XYX2Y2)
            pred_bbox.append(box)
            image_bboxes.addBoundingBox(box)
            
        evaluator = scoring.Evaluator.Evaluator()
        preds = evaluator.GetPascalVOCMetrics(image_bboxes,
                                                IOUThreshold=iou_thr,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
            
        for label_dict in preds:
            label_dict.update({'true_positive': label_dict['total TP'],
                               'false_positive': label_dict['total FP'],
                               'false_negative': label_dict['total positives'] - label_dict['total TP']})
            del label_dict['total TP']
            del label_dict['total FP']
            label_dict.update({'p': label_dict['true_positive'] / \
            ((label_dict['true_positive'] + label_dict['false_positive'])),
            'r': label_dict['true_positive'] / \
            ((label_dict['true_positive'] + label_dict['false_negative']))})
            label_dict.update({'F1': 2 * \
            (label_dict['p'] * label_dict['r']) / \
            (label_dict['p'] + label_dict['r'])})
        
        
        return preds
    
    
    def get_model_metrics(self, predictions, iou_thr):
        """
        Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

        Parameters:
                
            iou_thr (float): value of IoU to consider as threshold for a
                true prediction.

        Returns:
            dict: true positives (int), false positives (int), false negatives (int)
        """
        
        image_bboxes = BoundingBoxes()
        gt_bbox=[]
        pred_bbox=[]
        for image in predictions.keys():
            
            for bbox in predictions[image]['gt_bbox']:
                box = BoundingBox(imageName=image, classId=bbox['class'], 
                                              x=bbox['xmin'], y=bbox['ymin'], 
                                              w=bbox['xmax'], h=bbox['ymax'], 
                                              typeCoordinates=CoordinatesType.Absolute,
                                              bbType=BBType.GroundTruth, 
                                              format=BBFormat.XYX2Y2)
                gt_bbox.append(box)
                image_bboxes.addBoundingBox(box)
            
            for bbox in predictions[image]['pred_bbox']:
                box = BoundingBox(imageName=image, classId=bbox['class'], 
                                              classConfidence=bbox['score'],
                                              x=bbox['xmin'], y=bbox['ymin'], 
                                              w=bbox['xmax'], h=bbox['ymax'],  
                                              typeCoordinates=CoordinatesType.Absolute,
                                              bbType=BBType.Detected, 
                                              format=BBFormat.XYX2Y2)
                pred_bbox.append(box)
                image_bboxes.addBoundingBox(box)
            
        evaluator = scoring.Evaluator.Evaluator()
        preds = evaluator.GetPascalVOCMetrics(image_bboxes,
                                                IOUThreshold=iou_thr,
                                                method=MethodAveragePrecision.EveryPointInterpolation)  
        
        for label_dict in preds:
            label_dict.update({'true_positive': label_dict['total TP'],
                               'false_positive': label_dict['total FP'],
                               'false_negative': label_dict['total positives'] - label_dict['total TP']})
            del label_dict['total TP']
            del label_dict['total FP']
            label_dict.update({'p': label_dict['true_positive'] / \
            ((label_dict['true_positive'] + label_dict['false_positive'])),
            'r': label_dict['true_positive'] / \
            ((label_dict['true_positive'] + label_dict['false_negative']))})
            label_dict.update({'F1': 2 * \
            (label_dict['p'] * label_dict['r']) / \
            (label_dict['p'] + label_dict['r'])})
        
        return preds