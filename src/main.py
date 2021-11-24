import warnings;
warnings.filterwarnings('ignore');

import os
import sys
from scoring.integrated_scoring import IntegratedScoring
from scoring.locate_packets import locate_packets, locate_packets_pickle
from utils.config import get_config
from processing.calculate_compliance import Compliance

config = get_config("production_config")

if __name__ == "__main__":

    print('Switched to 2nd Docker')

    # try:
    #     print('Scoring Starts - Sub Brand Detection')
    #     scoring_obj = IntegratedScoring(curr_model= 'sub_brand_detection', index_lower=None, index_upper=None)
    #     scoring_obj.scoring()
    #     print('Scoring Ends - Sub Brand Detection')
    # except Exception as e:
    #     print(e)
    #     print("Unable to run Sub-brand detection.")
    #     sys.exit(1)
        
    # try:
    #     print('Model Integration Starts')
    #     scoring_obj = IntegratedScoring(curr_model= 'integration', index_lower=None, index_upper=None)
    #     scoring_obj.scoring()
    #     print('Model Integration Completes')
    # except Exception as e:
    #     print(e)
    #     print("Unable to integrate model results.")
    #     sys.exit(1)
    
    try:
        print("Locating packet positions.")    
        pred_dir = os.path.join(config['data_path']['blob_base_dir'],
                                 config['integrated_output']['integrated_dir'],
                                 config['data_path']['output_images_folder'])
        predictions = os.listdir(pred_dir)
        
        if len(predictions) < 1:
            print("No predictions available.")
            sys.exit(1)
        
        location_dict = locate_packets(pred_dir, predictions)
        print("Packet positions found.") 
    except Exception as e:
        print(e)
        print("Unable to find packet locations.")
        sys.exit(1)
    
    try:
        comp_obj = Compliance(location_dict)
        comp_obj.calculate_compliance(images_folder=None,
                                      model_output_folder=None)
    except Exception as e:
        print(e)
        print("Unable to calculate compliance.")
        sys.exit(1)
