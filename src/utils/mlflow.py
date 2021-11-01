# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:04:45 2020

@author: rajat.bansal
"""

import warnings;
warnings.filterwarnings('ignore');

import mlflow
import os, os.path


class MLFlow:
    
    def __init__(self, experiment_name, artifacts_location, tracking_uri, temp_artifacts_folder, run_name= None,
                delete_existing= False):
        '''
        experiment_name: use same experiment name to club multiple runs
        run_name: run name (keep unique across different iterations)
        artifacts_location: location to save the images and other files
        tracking_uri: location to save the meta data of runs
        temp_artifacts_folder: mlflow needs to save images before the artifacts to a run (use this folder to temp save the images)
                                this folder should be cleared for every new run
        '''
        
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.artifacts_location = artifacts_location
        self.tracking_uri = tracking_uri
        self.temp_artifacts_folder = temp_artifacts_folder
        
        if delete_existing:
            for dirpath, dirnames, filenames in os.walk(self.temp_artifacts_folder):
                if filenames:
                    for file in filenames:
                        if file:
                            os.remove(os.path.join(dirpath, file))
        
    
    def define_session(self):
        '''
        initialize the MLFlow experiment and run
        '''
        
        mlflow.set_tracking_uri(self.tracking_uri)
         
        try:
            mlflow.create_experiment(self.experiment_name, artifact_location= self.artifacts_location)
        except:
            print('Experiment name already exists')

        print(self.experiment_name)
        print(self.artifacts_location)
        print(mlflow.get_experiment_by_name(self.experiment_name))
        
        self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        
        if self.run_name:
            mlflow.start_run(experiment_id= self.experiment_id, run_name= self.run_name)
        else:
            mlflow.start_run(experiment_id= self.experiment_id)
        
        
    def end_session(self):
        '''
        stop the ML flow session
        '''
        
        mlflow.end_run()
        
        
    def log_metrics(self, metrics_dict):
        '''
        logging of the metrics. values should be numeric
        '''
        mlflow.log_metrics(metrics_dict)
        
        
    def log_params(self, params_dict):
        '''
        logging the parameters.
        '''
        
        mlflow.log_params(params_dict)
        
        
    def log_artifact(self, delete_existing= False):
        '''
        deletes the existing folder to ensure that images are not replicated across different artifacts
        '''
                            
        mlflow.log_artifact(self.temp_artifacts_folder)
        
        if delete_existing:
            for dirpath, dirnames, filenames in os.walk(self.temp_artifacts_folder):
                if filenames:
                    for file in filenames:
                        if file:
                            os.remove(os.path.join(dirpath, file))
                            
    
    def log_model_metrics(self, model_metrics, save_folder):
        rename_dict = {'class': 'Class', 'total positives': 'Total Positives', 'true_positive': 'true_positive',
                       'false_positive': 'false_positive', 'false_negative': 'false_negative', 'p': 'precision',
                       'r': 'recall', 'F1': 'F1'}
        model_metrics = model_metrics.rename(columns = rename_dict)
        model_metrics.to_csv(self.temp_artifacts_folder + save_folder + '/results.csv', index= False)
        
        for i in model_metrics.iterrows():
            metric = i[1].to_dict()
            class_name = metric['Class']
            
            if save_folder == 'valid':
                prefix = 'Validation-'
            elif save_folder == 'train':
                prefix = 'Training-'
            else:
                pass
             
            if model_metrics.shape[0] > 1:
                metric = {prefix + key + "-" + class_name.replace("||", "-") : value for key, value in metric.items() if key in ['precision', 'recall', 'F1']}
            else:
                metric = {prefix + key : value for key, value in metric.items() if key in ['precision', 'recall', 'F1', 'false_negative', 'false_positive', 'true_positive']}
            self.log_metrics(metric)
                        
        