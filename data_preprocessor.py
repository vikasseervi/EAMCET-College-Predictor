# data_preprocessor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from util import BRANCH

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.Y = None

    def encode_features(self):
        self.data['gender'] = self.data['gender'].map({'F': 0, 'M': 1})
        self.data['caste'] = self.data['caste'].map({'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8})
        self.data['branch_code'] = self.data['branch_code'].map(lambda x: BRANCH[x][0])
        return self.data

    def separate_features_labels(self, target_column, selected_features=None):
        if selected_features:
            self.X = self.data[selected_features]
        else:
            self.X = self.data.drop(target_column, axis=1)
        self.Y = self.data[target_column]
        return self.X, self.Y

    def split_data(self, test_size=0.6, random_state=3):
        return train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)
