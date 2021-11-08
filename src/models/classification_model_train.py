from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from utils.config import get_config
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import pickle
import os

class ClassificationModel:
    '''Train the classification model using Pre-Trained models'''
    
    def __init__(self):
        
        self.config = get_config("classification_crop")
        
        # input data
        self.input_data_directory = self.config['classification_model_training']['input_data_directory']
        self.test_directory = self.config['classification_model_training']['test_directory']
        
        # image processing
        self.image_size = self.config['classification_model_training']['image_size']
        self.width_shift_range = self.config['classification_model_training']['width_shift_range']
        self.height_shift_range = self.config['classification_model_training']['height_shift_range']
        self.shear_range = self.config['classification_model_training']['shear_range']
        self.zoom_range = self.config['classification_model_training']['zoom_range']
        
        # model architecture
        self.pretrained_base = self.config['classification_model_training']['pretrained_base']
        self.train_layers = self.config['classification_model_training']['train_layers']
        self.train_batchnormalization = self.config['classification_model_training']['train_batchnormalization']
        self.stack_layers = self.config['classification_model_training']['stack_layers']
        self.n_classes = self.config['classification_model_training']['n_classes']
        
        # model training
        self.learning_rate = self.config['classification_model_training']['learning_rate']      
        self.loss = self.config['classification_model_training']['loss']
        self.epochs = self.config['classification_model_training']['epochs']
        self.class_weight_lower = self.config['classification_model_training']['class_weight_lower']
        self.class_weight_upper = self.config['classification_model_training']['class_weight_upper']
        self.batch_size = self.config['classification_model_training']['batch_size']
        self.use_callback_red_learning_rate = self.config['classification_model_training']['use_callback_red_learning_rate']
        self.use_callback_early_stopping = self.config['classification_model_training']['use_callback_early_stopping']
        
        # model save path
        self.save_base_dir = self.config['classification_model_training']['save_base_dir']
        
        self.train_directory = self.input_data_directory + 'train/'
        self.valid_directory = self.input_data_directory + 'valid/'
        
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        self.callbacks = []
        
        if self.use_callback_red_learning_rate:
            self.callbacks.append(self.reduce_lr)
            
        if self.use_callback_early_stopping:
            self.callbacks.append(self.early_stopping)
            
        self.data_generators()
        self.calculate_class_weights()
        self.load_model()
            
        
        from datetime import date
        today = date.today()
        date = today.strftime("%m%d")
        
        self.model_name = f'{date}_{self.pretrained_base}_{self.learning_rate}LR_{self.epochs}Epoch_{self.n_classes}Classes/'
        
        self.model_save_path = self.save_base_dir + self.model_name
        
        try:
            os.mkdir(self.model_save_path)
        except e:
            print("Can't make model path: ", e)
            pass
        
        # save data dictionary
        model_params = open(self.model_save_path + "model_params.txt","w")
        for key, value in self.config['classification_model_training'].items():
            model_params.write(str(key) + " : " + str(value) + "\n")
        model_params.close()
            
    def data_generators(self):
        '''Creates the data generators to load train/valid/test images'''
        
        self.train_datagen = ImageDataGenerator(
                rescale=1/255.,
                width_shift_range=self.width_shift_range,
                height_shift_range=self.height_shift_range,
                shear_range=self.shear_range,
                zoom_range=self.zoom_range,
                fill_mode='nearest')

        self.valid_datagen = ImageDataGenerator(
                rescale=1/255.,
                fill_mode='nearest')
        
        self.test_datagen = ImageDataGenerator(
                rescale=1/255.,
                fill_mode='nearest')
        
        self.train_generator = self.train_datagen.flow_from_directory(
                self.train_directory,
                target_size=(self.image_size,self.image_size),
                batch_size=self.batch_size)
        
        self.valid_generator = self.valid_datagen.flow_from_directory(
                self.valid_directory,
                target_size=(self.image_size,self.image_size),
                batch_size=self.batch_size)
        
        # sanity checks
        assert self.train_generator.class_indices == self.valid_generator.class_indices, "Mismatch in train validation indices"
        assert len(self.train_generator.class_indices) == self.n_classes, "Mismatch in no. of classes"
        
        
    def calculate_class_weights(self):
        self.training_distribution = {}
        total_training_bbox = 0
        
        for i in self.train_generator.filenames:
            total_training_bbox += 1
            sub_brand = i.split("/")[0]
            if sub_brand in self.training_distribution.keys():
                self.training_distribution[sub_brand] += 1
            else:
                self.training_distribution[sub_brand] = 1
        
        self.class_weight_dict = {}
        for key, value in self.training_distribution.items():
            self.class_weight_dict[self.train_generator.class_indices[key]] = np.clip(1/(100 * value/total_training_bbox),
                                  self.class_weight_lower, self.class_weight_upper)
            
            
    def load_model(self):
        if self.pretrained_base == 'VGG16':
            self.conv_base = VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=(self.image_size,self.image_size,3))
            
        elif self.pretrained_base == 'InceptionResNetV2':
            self.conv_base = InceptionResNetV2(weights='imagenet',
                                               include_top=False,
                                               input_shape=(self.image_size,self.image_size,3))
            
        elif self.pretrained_base == 'ResNet50':
            self.conv_base = ResNet50(weights='imagenet',
                                               include_top=False,
                                               input_shape=(self.image_size,self.image_size,3))
        
        else:
            assert False, 'Model not added'
            
        
        # freezing the layers
        self.conv_base.trainable = True
    
        set_trainable = False
        for layer in self.conv_base.layers:
            
            if layer.name in self.train_layers:
                set_trainable = True
                
            if self.train_batchnormalization:
                if "BatchNormalization" in layer.__class__.__name__:
                    set_trainable = True
            
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        
        self.model = models.Sequential()
        self.model.add(self.conv_base)
        self.model.add(layers.Flatten())
        
        for i in self.stack_layers:
            self.model.add(layers.Dense(i, activation='relu'))
            
        self.model.add(layers.Dense(self.n_classes, activation='softmax'))
        
        print(self.model.summary())
        
        
        self.model.compile(loss=self.loss,
                      optimizer=optimizers.RMSprop(lr=self.learning_rate),
                      metrics=['acc'])
        
        
    def train_model(self):
        self.history = self.model.fit(
                self.train_generator,
                steps_per_epoch=self.train_generator.samples//self.batch_size,
                epochs=self.epochs,
                validation_data=self.valid_generator,
                validation_steps=self.valid_generator.samples//self.batch_size,
                callbacks=self.callbacks)

        print('Training complete')

        self.model.save(self.model_save_path + 'model.h5')
        with open(self.model_save_path + 'class_indices.pickle', 'wb') as handle:
            pickle.dump(self.train_generator.class_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

                
    def create_evaluation_metrics(self, image_files, base_dir, class_indices_dict):
        '''Returns the evaulation table with precision, recall & f1-Score
        Params:
            image_files (list): images with naming convention to include sub-brand label
            base_dir (string): directory having images in structure sub_brand --> images
        '''
        
        # dictionary to include metrics of all the classes
        final_dict = {}
    
        for i in tqdm(image_files):
            i = base_dir + i
            
            # to be replaced with keras data generator to make batch predictions
            image = Image.open(i)
            
            image = image.resize(size= (self.image_size, self.image_size))
            image = np.array(image)/255.0
            image = image.reshape(1,self.image_size,self.image_size,3)
    
            actual_label = i.split("/")[-2]
            pred_label = class_indices_dict[np.argmax(self.model.predict(image))]
    
            # initializing dict for each sub-brand
            if actual_label not in final_dict.keys():
                final_dict[actual_label] = {'Total Positive': 0, 'TP': 0, 'FP': 0, 'FN': 0}
    
            if pred_label not in final_dict.keys():
                final_dict[pred_label] = {'Total Positive': 0, 'TP': 0, 'FP': 0, 'FN': 0}
    
            # increment the values
            final_dict[actual_label]['Total Positive'] += 1
    
            if actual_label == pred_label:
                final_dict[actual_label]['TP'] += 1
            else:
                final_dict[pred_label]['FP'] += 1
                final_dict[actual_label]['FN'] += 1
                
                
        df_results = pd.DataFrame(columns = ['Sub-Brand', 'Total Positives', 'TP', 'FP', 'FN'])
    
        for key, value in final_dict.items():
            row = [key, value['Total Positive'], value ['TP'], value['FP'], value['FN']]
            a_series = pd.Series(row, index = df_results.columns)
            df_results = df_results.append(a_series, ignore_index=True)
        
        # calculation of precision & recall
        r_num = np.where(df_results['Total Positives'] == 0, 0, 100 * df_results['TP'])
        r_denom = np.where(df_results['Total Positives'] == 0, 1, df_results['Total Positives'])
        df_results['Recall'] =  r_num/r_denom
    
        p_denom = np.where(df_results['TP'] + df_results['FP'] == 0, 1, df_results['TP'] + df_results['FP'])
        df_results['Precision'] = 100 * df_results['TP']/p_denom
        
        p = df_results['Precision']
        r = df_results['Recall']
        df_results['F1-Score'] = (2 * p * r)/np.where((p + r) == 0, 1, (p + r))
         
        return df_results
    
    def evaluate_model(self):
        class_indices_dict = {value:key for key,value in self.train_generator.class_indices.items()}
        
        # validation metrics
        validation_df = self.create_evaluation_metrics(self.valid_generator.filenames, 
                                                       base_dir=self.valid_directory, 
                                                       class_indices_dict=class_indices_dict)
        
        # testing metrics
        all_sub_brands = os.listdir(self.test_directory)

        all_files = []
        for sub_brand in all_sub_brands:
            for image in os.listdir(self.test_directory + sub_brand):
                all_files.append(sub_brand + "/" + image)
                
        testing_df = self.create_evaluation_metrics(all_files,
                                                    base_dir=self.test_directory,
                                                    class_indices_dict=class_indices_dict)
        
        validation_df.to_csv(self.model_save_path + 'validation.csv', index= False)
        testing_df.to_csv(self.model_save_path + 'testing.csv', index= False)
        

    def cross_category_evaluation(self):  

        class_indices_dict = {value:key for key,value in self.train_generator.class_indices.items()}
        test_sub_brands = [v for v in class_indices_dict.values() if v in os.listdir(self.test_directory)]
                           
        df_cross_eval = pd.DataFrame(index= test_sub_brands, columns= test_sub_brands)
        df_cross_eval = df_cross_eval.fillna(0)
        # df_cross_eval.loc['Cheetos||Bolitas', 'Cheetos||Bolitas'] = 10
        
        from tensorflow.keras.preprocessing import image

        for value in tqdm(test_sub_brands):
            sub_brand_folder = self.test_directory + value + "/"
        
            # load all images into a list
            images = []
            for img in os.listdir(sub_brand_folder):
                img = os.path.join(sub_brand_folder, img)
                img = image.load_img(img, target_size=(self.image_size, self.image_size))
                img = image.img_to_array(img)
                img = img/255.0
                img = np.expand_dims(img, axis=0)
                images.append(img)
        
            images = np.vstack(images)
            classes = self.model.predict_classes(images, batch_size=10)
            classes = [class_indices_dict[i] for i in classes]
            for i in classes:
                df_cross_eval.loc[value, i] += 1
                
        row_sum = df_cross_eval.sum(axis= 1)
        for col in df_cross_eval.columns:
            df_cross_eval[col] = df_cross_eval[col]/row_sum
            
        df_cross_eval.to_csv(self.model_save_path + 'cross_category_evaluation_test.csv')
                                
        
    def train_and_evaluate(self):
        self.train_model()
        self.evaluate_model()
        self.cross_category_evaluation()