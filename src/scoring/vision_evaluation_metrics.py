def get_iou(json, xml_annotation_file):
    '''
    JSON (predicted) : N x 4 array of bounding boxes (x1, y1, x2, y2)
    Annotations (truth) : M x 4 array of bounding boxes (x1, y1, x2, y2)

    Return: IoU N x M array, pair wise IoU between each predicted and truth
    '''

