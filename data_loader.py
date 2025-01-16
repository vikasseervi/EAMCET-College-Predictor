# data_loader.py
import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.college_dataset = None

    def load_data(self):
        self.college_dataset = pd.read_csv(self.file_path)
        return self.college_dataset

    def display_info(self):
        print(self.college_dataset.shape)
        print(self.college_dataset.head())
        print(self.college_dataset.describe())

    def check_missing_values(self):
        return self.college_dataset.isnull().sum()
