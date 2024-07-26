import matplotlib as plt
import numpy as np
import os
import sys
sys.path.append('./src')
import logger
import file_operatons
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import KNNImputer
from utils import CSLSum
import file_operatons
import yaml
import shutil
import glob


class TrainDataPipe:

    """
       Module to perform data preprocessing on train data.
       Includes feature-engineering, handling missing values, transformation and scaling
       We save a pipeline model at the end, so that we can use it during prediction
       """

    def __init__(self, path):
        self.path = path
        with open('./config/FeatureEng.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        self.logger = logger.create_logger('TraintDataPipeline')
    def delete_path(self):

        try:
            if os.path.exists(self.config['processed_predict_data_path']):
                shutil.rmtree(self.config['processed_predict_data_path'])


        except Exception as e:
            self.logger.info(str(e))
            raise e

    def train(self):

        try:

            self.logger.info("Training Process Started!!")

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

            data[self.config['target_col']] = data[self.config['target_col']].apply(lambda x: 1 if x == 'Y' else 0)

            target_col = data[self.config['target_col']]

            data.drop(columns=[self.config['target_col']],inplace=True)

            ord_cat_cols_pipe = Pipeline(
                [("impute", SimpleImputer(missing_values=np.nan, fill_value='Other', strategy='constant')),
                 ("ordenc", OrdinalEncoder())])


            ohe_cat_cols_pipe = Pipeline(
                [("impute", SimpleImputer(missing_values=np.nan, fill_value='Other', strategy='constant')),
                 ("ohenc", OneHotEncoder())])

            num_cols = list(num_df.columns)
            num_cols_pipe = Pipeline([("knnimpute", KNNImputer(n_neighbors=5)),
                                      ("stdScaler", StandardScaler())])
            csl_pipe = Pipeline([("cslpipe", CSLSum()), ("stdScaler", StandardScaler())])

            self.preprocessing = ColumnTransformer([
                ("ord_cat_cols_pipe", ord_cat_cols_pipe, self.config['ord_categorical_cols']),
                ("ohe_cat_cols_pipe", ohe_cat_cols_pipe, self.config['ohe_categorical_cols']),
                ("csl_sum", csl_pipe, [self.config['policy_csl']]),
                ("num_pipe", num_cols_pipe, num_cols)
            ], remainder='passthrough', verbose_feature_names_out=True)

            transformed_data = pd.DataFrame(self.preprocessing.fit_transform(data), columns=self.preprocessing.get_feature_names_out())
            self.fileops = file_operatons.FileOperation()
            self.fileops.save_model(self.preprocessing, 'preprocessing-pipeline')

            transformed_data_concat = pd.concat([transformed_data, target_col], axis=1)
            os.makedirs(self.config['processed_train_data_path'],exist_ok=True)
            transformed_data_concat.to_csv(self.config['processed_train_data_path'] + '/transformed_train_data.csv', index=True)

            self.logger.info("Training Process Completed!!")

            return transformed_data_concat.iloc[:,:-1], transformed_data_concat.iloc[:,-1]

        except Exception as e:
            self.logger.info(str(e))
            raise   e


if __name__ == '__main__':
    path= '../insuranceFraud.csv'
    train_pipe= TrainDataPipe(path)
    train_pipe.train()
    train_pipe.delete_path()



