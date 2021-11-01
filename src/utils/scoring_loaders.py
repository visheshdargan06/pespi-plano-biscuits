# TODO: implement scoring loaders


import os
from PIL import Image
from utils.config import get_config
import json

# function to filter image using index (technically select a range of images from a folder)
def filter_images(all_images, index_lower, index_upper):
    '''filter the images on the basis of lower & upper indices'''

    if (index_lower is not None) & (index_upper is not None):
        all_images = all_images[index_lower:index_upper]
    elif (index_lower is not None) & (index_upper is None):
        all_images = all_images[index_lower:]
    elif (index_lower is None) & (index_upper is not None):
        all_images = all_images[:index_upper]
    else:
        pass
    
    return all_images

# image loader function that yields the images in a folder
    # load paths
    # iterate over the images in folder
    # check if that's an image
    # filter the desired images
    # open image using PIL and yield it\

def image_loader(folder_path= None, index_lower= None, index_upper= None):
    '''yields the images in the folder'''
    
    if not folder_path:
        config = get_config("production_config")
        base_dir = config['data_path']['blob_base_dir']
        folder_path = base_dir + config['data_path']['images_dir']
    
    all_images = os.listdir(folder_path)
    all_images = [i for i in all_images if 'ipynb' not in i]
    all_images = [i for i in all_images if 
                  (i.endswith('.png') | i.endswith('.jpeg') | i.endswith('.jpg') | 
                          i.endswith('.PNG') | i.endswith('.JPEG') | i.endswith('.JPG'))]
    all_images = filter_images(all_images=all_images, index_lower=index_lower, index_upper=index_upper)
    
    for i in all_images:
        image_path = os.path.join(folder_path, i)
        image = Image.open(image_path)
        width, height = image.size
        yield (i, image, (width, height))

# intermediate json - function to save a json in specified path
def intermediate_json(p_dict, file_path= None):
    '''saves the intermediate dictionary as json file to the specified folder'''
    
    with open(file_path, 'w') as json_file:
        json.dump(p_dict, json_file)

# function to save final json in specified path
def final_json(p_dict, json_name):
    '''saving the final integrated json file'''
    
    config = get_config("production_config")
    base_dir = config['data_path']['blob_base_dir']
    output_images_folder = config['data_path']['output_images_folder']
    
    config = get_config("production_config")
    integrated_dir = base_dir + config['integrated_output']['integrated_dir']
    integrated_dir = integrated_dir + output_images_folder
    try:
        os.mkdir(integrated_dir)
    except:
        pass
    
    with open(integrated_dir + json_name, 'w') as json_file:
        json.dump(p_dict, json_file) 