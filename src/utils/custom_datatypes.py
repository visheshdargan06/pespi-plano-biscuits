import numpy as np

class Row:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def __str__(self):
        return 'Bounding Box: ({}, {}, {}, {})'.format(self.x1, 
                                                     self.y1,
                                                     self.x2,
                                                     self.y2)
    def __repr__(self):
        return '({}, {}, {}, {})'.format(self.x1, 
                                         self.y1,
                                         self.x2,
                                         self.y2)
    def get_bbox(self):
        return np.array([self.x1, 
                         self.y1,
                         self.x2,
                         self.y2])
    
class Packet(Row):
    def __init__(self, label, x1, y1, x2, y2):
        self.label = label
        Row.__init__(self, x1, y1, x2, y2)
        
    def __str__(self):
        return 'Subbrand: {}, Bounding Box: ({}, {}, {}, {})'.format(self.label, 
                                                                     self.x1, 
                                                                     self.y1,
                                                                     self.x2,
                                                                     self.y2)
    
    def get_label(self):
        return self.label


class CompleteRack:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def __str__(self):
        return 'Bounding Box: ({}, {}, {}, {})'.format(self.x1, 
                                                     self.y1,
                                                     self.x2,
                                                     self.y2)
    def __repr__(self):
        return '({}, {}, {}, {})'.format(self.x1, 
                                         self.y1,
                                         self.x2,
                                         self.y2)
    def get_bbox(self):
        return np.array([self.x1, 
                         self.y1,
                         self.x2,
                         self.y2])