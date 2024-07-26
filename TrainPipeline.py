import os
import sys
import logger
from src import RawDataValidation
from src import TrainDataPipeline
from src import ModelTrainer
import yaml

class TrainingPipeline:

    def __init__(self,data_path=None):
        self.logger = logger.create_logger('TrainPipeline')
        self.data_path = data_path
        with open('./config/TrainPipeline.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        if self.data_path  is None:
            self.data_path  = self.config['train_raw_data_path']
        self.raw_data_validation = RawDataValidation.TrainRawDataValidationPipeline(self.data_path)
        self.train_data_pipeline = TrainDataPipeline.TrainDataPipe(self.config['train_validated_data'])
        self.model_trainer = ModelTrainer.ModelTrain()


    def train_pipe(self):
        self.logger.info("Training Pipeline Initiated")
        self.logger.info("Raw Data Validation Initiated")
        self.raw_data_validation.raw_validation_pipe()
        self.logger.info("Raw Data Validation Completed Successfully!!")
        self.logger.info("Train Data Pipeline Initiated!!")
        self.train_data_pipeline.delete_path()
        X_train, y_train = self.train_data_pipeline.train()
        self.best_model = self.model_trainer.get_best_model(X_train,y_train, X_train, y_train)     # Change Second set to Test Data. Add it to the pipeline


if __name__=='__main__':

    pipe= TrainingPipeline()
    pipe.train_pipe()








