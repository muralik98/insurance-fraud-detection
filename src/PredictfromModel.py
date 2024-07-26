import matplotlib as plt
import numpy as np
import os
import sys
sys.path.append('./src')
import logger
import file_operatons
import pandas as pd
import yaml
import shutil
import glob


class PredictOutput:
    """
    Module used to test predict-data results.
    We load saved model during training.
    """

    def __init__(self):
        with open('./config/PredictPipeline.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        self.logger = logger.create_logger('Predict From Model')

    def predict_from_model(self, X_predict):

        try:
            self.logger.info("Predict From Model Started")
            fileOp = file_operatons.FileOperation()
            self.saved_model = fileOp.load_model('best-model')

            self.y_predicted = self.saved_model.predict(X_predict)  # Predictions using the XGBoost Model
            self.y_preddicted_df = pd.DataFrame(self.y_predicted,columns=['Output'])

            os.makedirs(self.config['final_predicted_output'], exist_ok=True)
            self.y_preddicted_df.to_csv(self.config['final_predicted_output'] + '/final_predicted_output.csv',
                                           index=True)
            self.logger.info("Predict From Model Completed Successfully!!! Output file saved in the location"+str(self.config['final_predicted_output']))

            return self.config['final_predicted_output'] + '/final_predicted_output.csv'

        except Exception as e:
            self.logger.info(str(e))
            raise e
