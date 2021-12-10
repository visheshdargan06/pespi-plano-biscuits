import pathlib
import subprocess
import sys
import glob
import shutil
import os
from utils.config import get_config



os.environ["AZCOPY_JOB_PLAN_LOCATION"]="/tf"
os.environ["AZCOPY_LOG_LOCATION"]="/tf"

test_blob = "https://pepsiplanobiscuits.blob.core.windows.net/common/model_files_planogram/planogram/data/processed?sp=racwl&st=2021-12-02T10:03:30Z&se=2021-12-02T18:03:30Z&sv=2020-08-04&sr=c&sig=rIrzZiyKtsD%2FZxkzkWVJoIQS9fkd5N88AiQWWFRiGCA%3D"

try:
    run_date = sys.argv[1]
    config = get_config("production_config")
    images_location = os.path.join(config.get('path').get('blob_base_dir'),config.get('data_path').get('images_dir'))
    pathlib.Path(images_location).mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["./azcopy/azcopy", 
                    "copy", 
                    "https://latamdev.blob.core.windows.net/latamimageproc/"+run_date+"/?sv=2019-10-10&st=2020-10-28T13%3A16%3A46Z&se=2020-11-29T13%3A16%3A00Z&sr=c&sp=racwl&sig=eWPyKJa0HG3xr%2BwYX8A6BF8hFtUKRBmYNWtcZHnuo%2F4%3D", 
                    images_location, 
                    "--recursive"])

    src_dir = images_location+run_date
    if (not os.path.isdir(src_dir)):
        sys.exit(100)
    files = os.listdir(src_dir)
    if len(files)==0:
        sys.exit(100)
    dst_dir = images_location
    for file in files:
        shutil.move(os.path.join(src_dir, file), os.path.join(dst_dir, file))
    shutil.rmtree(src_dir)
except Exception as e:
    print(e)
    sys.exit(101)
sys.exit(0)

