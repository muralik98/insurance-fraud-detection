import pickle
import os
import shutil
import sys
sys.path.append('./src')
import logger
import yaml

class FileOperation:
    """

    Class used to perform saving of model and loading model
    """

    def __init__(self):
        self.logger = logger.create_logger('File_Operations')
        with open('./config/FileOps.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def save_model(self, model, filename):

        try:
            path = os.path.join(self.config['model_path'], filename)
            if os.path.isdir(path):
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path+'/'+filename+'.sav','wb') as f:
                pickle.dump(model, f)

        except Exception as e:
            self.logger.info('Error While Validating Filenames!'+ '||' + str(e))
            raise Exception()

    def load_model(self, filename):

        try:
            with open(self.config['model_path'] + '/' + str(filename) + '/' + str(filename) + '.sav', 'rb') as f:
                return pickle.load(f)

        except Exception as e:
            self.logger.info('Error While Validating Filenames!'+ '||' + str(e))
            raise Exception()


