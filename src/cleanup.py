import sys
import os
import shutil
from utils.config import get_config

config = get_config("production_config")
images_location = os.path.join(config.get('path').get('blob_base_dir'),config.get('data_path').get('images_dir'))
for filename in os.listdir(images_location):
    file_path = os.path.join(images_location, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        sys.exit(1)
        print('Failed to delete %s. Reason: %s' % (file_path, e))
        
