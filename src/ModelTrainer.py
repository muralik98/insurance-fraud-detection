import matplotlib as plt
import numpy as np
import os
import sys
sys.path.append('./dataPreprocessing')
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score,accuracy_score
from xgboost import XGBClassifier
import logger
import yaml
import file_operatons

class ModelTrain:
    """
    Module to train the model using train data.
    We use grid-search to find best parameter.
    We use Xg-Boost which was found to be optimal in EDA.
    We can also add other models to check performance. This can be done easly by creating dictionary of models and storing parameters,
    which can be evaluated for best model
    """

    def __init__(self):
        self.logger = logger.create_logger('ModelTrainer')
        with open('./config/ModelTrain.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def get_best_params_for_xgboost(self, X_train, y_train):

        self.logger.info('Finding Best Parameter for XgBoost')
        try:

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss'), self.config['param_grid_xgboost'], verbose=3,
                                     cv=5)
            # finding the best parameters
            self.grid.fit(X_train, y_train)

            # extracting the best parameters

            # Extracting the best parameters
            self.best_params = self.grid.best_params_
            max_depth = self.best_params['max_depth']
            n_estimators = self.best_params['n_estimators']

            # Creating a new model with the best parameters
            self.xgb = XGBClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            # training the mew model
            self.xgb.fit(X_train, y_train)
            self.logger.info('Finding Best Parameter for XgBoost'+'XGBoost best params: ' + str(self.grid.best_params_))
            return self.xgb
        except Exception as e:
            self.logger.info('Error in Training'+str(e))
            raise Exception()

    def get_best_model(self,X_train,y_train,X_test,y_test):

        self.logger.info('Finding Best Model')

        try:
            self.xgboost= self.get_best_params_for_xgboost(X_train,y_train)
            self.prediction_xgboost = self.xgboost.predict(X_test) # Predictions using the XGBoost Model

            os.makedirs('./config/ModelParamOut/', exist_ok=True)
            with open('./config/ModelParamOut/param_grid_xgboost.yaml', 'w') as file:
                yaml.dump(self.xgb.get_params(), file)

            if len(y_test.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(y_test, self.prediction_xgboost)
                self.logger.info('Accuracy for XGBoost:'+str(self.xgboost_score ))
            else:
                #self.xgboost_score = roc_auc_score(y_test, self.prediction_xgboost) # AUC for XGBoost
                self.xgboost_score = accuracy_score(y_test, self.prediction_xgboost)
                self.logger.info('Accuracy for XGBoost:'+str(self.xgboost_score ))

            self.fileops = file_operatons.FileOperation()
            self.fileops.save_model(self.xgboost, 'best-model')

            return 'XGBoost',self.xgboost

        except Exception as e:
            self.logger.info('Error in Execution',e)
            raise Exception()

if __name__=='__main__':
    # Pilot code to test module
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

    train_pipe = ModelTrain()
    import pandas as pd
    X_train,X_test,y_train,y_test = pd.DataFrame(X), pd.DataFrame(X), pd.Series(y), pd.Series(y)
    model  = train_pipe.get_best_model(X_train,y_train,X_test,y_test)

