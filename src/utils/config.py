import sys
sys.path.append("../")
from utils.path import get_project_path
import yaml
import os

def get_config(file = "cpu_config"):
    """
    Get the config from the specified config file.
    
    Parameters:
    file (str): Name of the config file
    
    Returns:
    obj: yaml object    
    """
    
    project_path = get_project_path()
    config_path = os.path.join(project_path, 
                                "config/" + file + ".yaml")
    with open(config_path) as file:
        config = yaml.full_load(file)
        
    return config
