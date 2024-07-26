import os
import sys
import logger
from src import RawDataValidation
from src import PredictDataPipeline
from src import PredictfromModel
import yaml

class PredictPipeline:

    def __init__(self, data_path=None):
        self.logger = logger.create_logger('TestPipeline')
        self.data_path = data_path
        with open('./config/PredictPipeline.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        if self.data_path  is None:
            self.data_path  = self.config['test_raw_data_path']
        self.raw_data_validation = RawDataValidation.TestRawDataValidationPipeline(self.data_path)
        self.predict_data_pipeline = PredictDataPipeline.PredictDataPipe(self.config['test_validated_data'])
        self.predict_output = PredictfromModel.PredictOutput()


    def predict_pipe(self):
        self.logger.info("Training Pipeline Initiated")
        self.logger.info("Raw Data Validation Initiated")
        self.raw_data_validation.raw_validation_pipe()
        self.logger.info("Raw Data Validation Completed Successfully!!")
        self.logger.info("Train Data Pipeline Initiated!!")
        self.predict_data_pipeline.delete_path()
        X_predict = self.predict_data_pipeline.transform()
        output_path = self.predict_output.predict_from_model(X_predict)    # Change Second set to Test Data. Add it to the pipeline
        return output_path


if __name__=='__main__':

    pipe= PredictPipeline()
    pipe.predict_pipe()