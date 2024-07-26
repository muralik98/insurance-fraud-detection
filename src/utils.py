from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re

# Custom Class to convert CSL value into numerical value. Combined Single Limit (CSL)
# CSL stands for Combined Single Limit, which is a provision in an insurance policy that limits coverage for
# all components of a claim to a single dollar amount. This includes property damage, bodily injury per person, and bodily injury per accident.


class CSLSum(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Nothing to do here

    def transform(self, X):
        pattern = re.compile(r'^\d+/\d+$')
        X['policy_csl'] = X['policy_csl'].apply(
            lambda x: int(x.split('/')[0]) + int(x.split('/')[1]) if pattern.match(str(x)) else 0)
        X[X['policy_csl'] == 0] = X['policy_csl'].mean()
        return X

    def get_feature_names_out(self, input_features=None):
        return ['policy_csl']