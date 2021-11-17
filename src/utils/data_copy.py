# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:32:07 2020

@author: rajat.bansal
"""

import tarfile
import os
import shutil
from utils.config import get_config


class DataCopy:
    
    def __init__(self):
        '''
        Movement of data from one storage to another after compressing the data
        Params:
            folder_to_compress_path: complete path of folder to be compressed
            folder_to_save_compressed_file: complete path to save the compressed folder
            folder_to_copy_compressed_file: complete path to copy the compressed folder (i.e. path of target storage)
        '''
        
        self.config = get_config("yolo_config")
        self.folder_to_compress_path = self.config['data_copy']['folder_to_compress_path']
        self.folder_to_save_compressed_file = self.config['data_copy']['folder_to_save_compressed_file']
        self.folder_to_copy_compressed_file = self.config['data_copy']['folder_to_copy_compressed_file']
    
    
    def compress_folder(self):
        '''
        compressed the folder to .tar format
        '''
        with tarfile.open(self.folder_to_save_compressed_file, "w:gz") as tar:
            tar.add(self.folder_to_compress_path, arcname=os.path.basename(self.folder_to_compress_path))
            
            
    def copy_folder(self):
        '''
        copies the folder from source to target destination
        '''
        shutil.copy(self.folder_to_save_compressed_file, self.folder_to_copy_compressed_file)
            
        
    def decompress_folder(self):
        '''
        decompresses the tar file
        '''
        file = tarfile.open(self.folder_to_copy_compressed_file)
        file.extractall()
        file.close