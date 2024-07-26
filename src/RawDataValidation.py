import re
import shutil
from datetime import datetime
import os
import sys
import pandas as pd
import logger
import json
sys.path.append('./src')
import yaml

class TrainRawDataValidationPipeline:
    """
    Module to perform basic validation on Raw Data provided
    """


    def __init__(self,path):
        self.data_src = path
        self.logger = logger.create_logger('Train-Raw_Data_Validation_Pipeline')
        #self.regex_schema = r'^fraudDetection_\d.*_\d{6}\.csv$'           # Regex Schema to be used
        with open('./ifd_schema/schema_training.json', 'r') as file:
            self.schema = json.load(file)
        with open('./config/RawValidConfig.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def raw_validation_pipe(self):


        self.data_src_validation()


    def artifacts_handler(self):

        try:
            path=os.path.join(self.config['artifacts']['train_valid'])
            os.makedirs(path,exist_ok=True)
            path = os.path.join(self.config['artifacts']['train_invalid'])
            os.makedirs(path, exist_ok=True)

        except Exception as e:
            self.logger.info('Error While Creating raw_data_artifacts!')

    def artifacts_delete(self,path):

        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                self.logger.info('Number of Columns Mismatch!!')
        except Exception as e:
            self.logger.info('Error While Deleting Folder!')
            raise Exception()


    def data_src_validation(self):

        self.artifacts_delete(self.config['artifacts']['train_invalid'])         # Delete both Path created in previous fail run
        self.artifacts_delete(self.config['artifacts']['train_valid'])
        self.artifacts_handler()

        try:
            list_of_files = [file for file in os.listdir(self.data_src)]

            for file in list_of_files:
                if (re.search(self.config['regex_schema'], file)):

                    df = pd.read_csv(self.data_src+'/'+file)

                    if df.shape[1]!=self.schema['NumberofColumns']:      # If validation fails move raw file to invalid folder and continue with next file

                        self.logger.info(str(file)+'--Number of Columns Mismatch!!')
                        shutil.move(self.data_src+'/'+file, self.config['artifacts']['train_invalid'])
                        continue


                    if len(df.columns[df.isna().all()].tolist())>0:      # If validation fails move raw file to invalid folder and continue with next file
                        self.logger.info(str(file)+'--Null Column Found!!')
                        shutil.move(self.data_src+'/'+file, self.config['artifacts']['train_invalid'])
                        continue



                    shutil.copy(self.data_src+'/'+file, self.config['artifacts']['train_valid'])
                    self.logger.info(str(file)+'--Source Raw File Moved to data_src/train/valid_raw folder!!')


                else:
                    self.logger.info(str(file)+'--Schema Name Mismatch!! Refer to Schema Policy')
                    shutil.move(self.data_src + '/' + file, self.config['artifacts']['train_invalid'])
                    continue

        except Exception as e:
            self.logger.info(str(file)+'--Error While Validating Filenames!')
            raise Exception()



class TestRawDataValidationPipeline:
    """
    Module to perform basic validation on Raw Data provided
    """


    def __init__(self,path):
        self.data_src = path
        self.logger = logger.create_logger('Test-Raw_Data_Validation_Pipeline')
        #self.regex_schema = r'^fraudDetection_\d.*_\d{6}\.csv$'           # Regex Schema to be used
        with open('./ifd_schema/schema_prediction.json', 'r') as file:
            self.schema = json.load(file)
        with open('./config/RawValidConfig.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def raw_validation_pipe(self):


        self.data_src_validation()


    def artifacts_handler(self):

        try:
            path=os.path.join(self.config['artifacts']['test_valid'])
            os.makedirs(path,exist_ok=True)
            path = os.path.join(self.config['artifacts']['test_invalid'])
            os.makedirs(path, exist_ok=True)

        except Exception as e:
            self.logger.info('Error While Creating raw_data_artifacts!')

    def artifacts_delete(self,path):

        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                self.logger.info('Number of Columns Mismatch!!')
        except Exception as e:
            self.logger.info('Error While Deleting Folder!')
            raise Exception()


    def data_src_validation(self):

        self.artifacts_delete(self.config['artifacts']['test_invalid'])         # Delete both Path created in previous fail run
        self.artifacts_delete(self.config['artifacts']['test_valid'])
        self.artifacts_handler()

        try:
            list_of_files = [file for file in os.listdir(self.data_src)]

            for file in list_of_files:
                if (re.search(self.config['predict_regex_schema'], file)):

                    df = pd.read_csv(self.data_src+'/'+file)

                    if df.shape[1]!=self.schema['NumberofColumns']:      # If validation fails move raw file to invalid folder and continue with next file

                        self.logger.info(str(file)+'--Number of Columns Mismatch!!')
                        shutil.move(self.data_src+'/'+file, self.config['artifacts']['test_invalid'])
                        continue


                    if len(df.columns[df.isna().all()].tolist())>0:      # If validation fails move raw file to invalid folder and continue with next file
                        self.logger.info(str(file)+'--Null Column Found!!')
                        shutil.move(self.data_src+'/'+file, self.config['artifacts']['test_invalid'])
                        continue



                    shutil.copy(self.data_src+'/'+file, self.config['artifacts']['test_valid'])
                    self.logger.info(str(file)+'--Source Raw File Moved to data_src/train/valid_raw folder!!')


                else:
                    self.logger.info(str(file)+'--Schema Name Mismatch!! Refer to Schema Policy')
                    shutil.move(self.data_src + '/' + file, self.config['artifacts']['test_invalid'])
                    continue

        except Exception as e:
            self.logger.info(str(file)+'--Error While Validating Filenames!')
            raise Exception()


if __name__=='__main__':
    train_path = '../train_data_src'
    test_path = '../test_data_src'
    import json
    TrainRawValid = TrainRawDataValidationPipeline(train_path)
    TrainRawValid.raw_validation_pipe()

    TestRawValid = TestRawDataValidationPipeline(test_path)
    TestRawValid.raw_validation_pipe()








