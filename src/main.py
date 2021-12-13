import warnings;
warnings.filterwarnings('ignore');

import os
import sys
from scoring.integrated_scoring import IntegratedScoring
from scoring.locate_packets import locate_packets
from utils.config import get_config
from processing.calculate_compliance import Compliance, ComplianceConsolidated
from scoring.check_image import image_sanity_check
config = get_config("production_config")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=False)
args=vars(ap.parse_args())
# try:
#     template_cond = args["mode"]  #All, OOS, default
# except:
#     template_cond = "default"

template_cond = 'all'
config = get_config("production_config")

if __name__ == "__main__":


    ####################################################################################################################
    #Detection of packets racks and rack-rows
    try:
        print('Scoring Starts - Packet Detection')
        scoring_obj = IntegratedScoring(curr_model= 'packets_detection', index_lower=None, index_upper=None)
        scoring_obj.scoring()
        print('Scoring Ends - Packet Detection')
    except Exception as e:
        print(e)
        print("Unable to perform packet detections.")
        sys.exit(1)

    try:
        print('Scoring Starts - Rack Row Detection')
        scoring_obj = IntegratedScoring(curr_model= 'rackrow_detection', index_lower=None, index_upper=None)
        scoring_obj.scoring()
        print('Scoring Ends - Rack Row Detection')
        
    except Exception as e:
        print(e)
        print("Unable to perform rack row detections.")
        sys.exit(1)


    ####################################################################################################################
    # Sanity check on images
    try: 
        print('Running Sanity Check on Images')
        image_sanity_check()
    except:
        print("Unable to perform sanity check on images.")


    ####################################################################################################################
    # Classification of packets into sub-brands
    try:
        print('Scoring Starts - Sub Brand Detection')
        scoring_obj = IntegratedScoring(curr_model= 'sub_brand_detection', index_lower=None, index_upper=None)
        scoring_obj.scoring()
        print('Scoring Ends - Sub Brand Detection')
    except Exception as e:
        print(e)
        print("Unable to run Sub-brand detection.")
        sys.exit(1)
        

    ####################################################################################################################
    # Integration of outputs
    try:
        print('Model Integration Starts')
        scoring_obj = IntegratedScoring(curr_model= 'integration', index_lower=None, index_upper=None)
        scoring_obj.scoring()
        print('Model Integration Completes')
    except Exception as e:
        print(e)
        print("Unable to integrate model results.")
        sys.exit(1)
    ####################################################################################################################

    try:
        print("Locating packet positions.")    
        pred_dir = os.path.join(config['path']['project'],
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



   
    
    '''    try:
            if template_cond == "Best" or template_cond == "best":
                comp_obj = ComplianceConsolidated(location_dict)
                comp_obj.calculate_compliance(all_comparissons=False)
            elif template_cond == "All" or template_cond == "all":
                comp_obj = ComplianceConsolidated(location_dict)
                comp_obj.calculate_compliance(all_comparissons=True)
            elif template_cond == "OoS" or template_cond == "oos":
                comp_obj = Compliance(location_dict)
                comp_obj.calculate_compliance(images_folder=None,
                                            model_output_folder=None,
                                            last_visit = True)
            else:
                comp_obj = Compliance(location_dict)
                comp_obj.calculate_compliance(images_folder=None,
                                            model_output_folder=None)
    '''
    ####################################################################################################################
    # Compliance calculation

    def merge_final_csvs(a,b):
        pass
    # try:
    # from utils.utils import merge_final_csvs
    final_compliance_save = os.path.join(config['path']['project'],
                                        config['compliance']['path']['output_prev_after_comparsion'])
    if template_cond == "All" or template_cond == "all":
        print("..... Processing 'All' mode compliance .....")
        comp_obj = ComplianceConsolidated(location_dict)
        comp_obj.calculate_compliance(all_comparissons=True)
        csv_list = ["detections_all.csv"]
        merge_final_csvs(final_compliance_save, csv_list)

    elif template_cond == "OOS" or template_cond == "oos" or template_cond == "OoS":
        # After
        print("..... Processing 'After' mode compliance .....")
        comp_obj = Compliance(location_dict)   
        comp_obj.calculate_compliance(images_folder=None,
                                    model_output_folder=None)
        # Best
        print("..... Processing 'Best' mode compliance .....")
        comp_obj = ComplianceConsolidated(location_dict)    
        comp_obj.calculate_compliance(all_comparissons=False)
        # Out of Stock
        print("..... Processing 'OOS' mode compliance .....")
        comp_obj = Compliance(location_dict)
        comp_obj.calculate_compliance(images_folder=None, 
                                    model_output_folder=None,
                                    last_visit = True)
        csv_list = ["detections_after.csv", "detections_best.csv", "OOS.csv"]
        merge_final_csvs(final_compliance_save, csv_list)

    elif template_cond == "After" or template_cond == "after":
        # After
        print("..... Processing 'After' mode compliance .....")
        comp_obj = Compliance(location_dict)   
        comp_obj.calculate_compliance(images_folder=None,
                                    model_output_folder=None)
        csv_list = ["detections_after.csv"]
        merge_final_csvs(final_compliance_save, csv_list)

    else:
        # After
        print("..... Processing 'After' mode compliance .....")
        comp_obj = Compliance(location_dict)   
        comp_obj.calculate_compliance(images_folder=None,
                                    model_output_folder=None)
        # Best
        print("..... Processing 'Best' mode compliance .....")
        comp_obj = ComplianceConsolidated(location_dict)    
        comp_obj.calculate_compliance(all_comparissons=False)

        csv_list = ["detections_after.csv", "detections_best.csv"]
        merge_final_csvs(final_compliance_save, csv_list)

    # except Exception as e:
    #     print(e)
    #     print("Unable to calculate compliance.")
    #     sys.exit(1)