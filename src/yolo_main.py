import warnings;
warnings.filterwarnings('ignore');

import os
import sys
from scoring.integrated_scoring import IntegratedScoring

if __name__ == "__main__":
    print('Scoring Starts...')

    try:
        print('Scoring Starts - Packet Detection')
        scoring_obj = IntegratedScoring(curr_model= 'packets_detection', index_lower=None, index_upper=None)
        scoring_obj.scoring()
        print('Scoring Ends - Packet Detection')
    except Exception as e:
        print(e)
        print("Unable to perform packet detections.")
        sys.exit(1)

    #try:
    print('Scoring Starts - Rack Row Detection')
    scoring_obj = IntegratedScoring(curr_model= 'rackrow_detection', index_lower=None, index_upper=None)
    scoring_obj.scoring()
    print('Scoring Ends - Rack Row Detection')
        
    # except Exception as e:
    #     print(e)
    #     print("Unable to perform rack row detections.")
    #     sys.exit(1)