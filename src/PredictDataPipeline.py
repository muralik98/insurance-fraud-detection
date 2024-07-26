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

class PredictDataPipe:

    """
    We perform pre-processing on the predict data.
    This includes feature engineering,scaling and handling missing values and outliers
    """

    def __init__(self, path):
        self.path = path
        with open('./config/FeatureEng.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        self.logger = logger.create_logger('PredictDataPipeline')

    def delete_path(self):

        try:
            self.logger.info("PredictDataPipeline Started")
            if os.path.exists(self.config['processed_predict_data_path']):
                shutil.rmtree(self.config['processed_predict_data_path'])

        except Exception as e :
            self.logger.info(str(e))
            raise e

    def transform(self):

        try:

            csv_files = glob.glob(os.path.join(self.path, "*.csv"))
            dataframes = []
            for file in csv_files:
                df = pd.read_csv(file)
                dataframes.append(df)
            data = pd.concat(dataframes, ignore_index=True)


            data = data.replace('?', np.nan)
            data.dropna(thresh=data.shape[1] // 3, inplace=True)


            data['active_days'] = abs(pd.to_datetime(data['policy_bind_date'], format='%m/%d/%Y') - pd.to_datetime(data['incident_date'],format='%m/%d/%Y')).dt.days

            data.drop(columns=self.config['col_to_delete'], inplace=True)

            num_df = data.select_dtypes(exclude=['object']).copy()
            cat_df = data.select_dtypes(include=['object']).copy()

            num_cols = list(num_df.columns)


            fileOp = file_operatons.FileOperation()
            self.saved_model = fileOp.load_model('preprocessing-pipeline')

            transformed_data = pd.DataFrame(self.saved_model.transform(data), columns=self.saved_model.get_feature_names_out())

            os.makedirs(self.config['processed_predict_data_path'],exist_ok=True)
            transformed_data.to_csv(self.config['processed_predict_data_path']+'/transformed_predict_data.csv', index=True)

            return transformed_data

        except Exception as e:
            self.logger.info(str(e))
            raise e


if __name__ == '__main__':

    # Pilot Code to test module

    path_test = '../InputFile.csv'

    train_pipe = PredictDataPipe(path_test)
    df = train_pipe.transform()
    #train_pipe.delete_path()


