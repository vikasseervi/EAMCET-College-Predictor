import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from util import BRANCH, COLLEGE


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


class DataVisualizer:
    @staticmethod
    def plot_category_distribution(data, category):
        sns.catplot(x=category, data=data, kind='count')
        plt.show()

    @staticmethod
    def plot_feature_vs_target(data, feature, target):
        plt.figure(figsize=(5, 5))
        sns.barplot(x=target, y=feature, data=data)
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data):
        numeric_data = data.select_dtypes(include=[np.number])
        correlation = numeric_data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')


class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.Y = None

    def encode_features(self):
        # Encode gender, caste, branch_code, branch, and college as categorical variables
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

    def split_data(self, test_size=0.2, random_state=3):
        return train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)


class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train_model(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def evaluate_model(self, X_test, Y_test):
        X_test_prediction = self.model.predict(X_test)
        return accuracy_score(X_test_prediction, Y_test)

#
# class Predictor:
#     def __init__(self, model, columns, original_data):
#         self.model = model
#         self.columns = columns
#         self.original_data = original_data
#
#     def predict(self, input_data):
#         input_data_as_numpy_array = np.asarray(input_data)
#         input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#         input_df = pd.DataFrame(input_data_reshaped, columns=self.columns)
#         prediction = self.model.predict(input_df)
#         return prediction[0]
#
#     def get_college_details(self, college_code):
#         college_row = self.original_data[self.original_data['college_code'] == college_code]
#         if not college_row.empty:
#             return college_row[['college_code', 'college', 'students_per_class', 'fee']].iloc[0]
#         else:
#             return None

class Predictor:
    def __init__(self, model, feature_columns, data):
        self.model = model
        self.feature_columns = feature_columns
        self.data = data  # Original dataset including college_code, college, students_per_class, fee

    def predict_college(self, input_data):
        # Convert the input list to a DataFrame for prediction
        input_df = pd.DataFrame([input_data], columns=self.feature_columns)

        # Predict the college code using the trained model
        predicted_college_code = self.model.predict(input_df)[0]

        # Look up the college, students_per_class, and fee using the predicted college code
        college_info = self.data[self.data['college_code'] == predicted_college_code].iloc[0]

        return {
            'college_code': predicted_college_code,
            'college': college_info['college'],
            'students_per_class': college_info['students_per_class'],
            'fee': college_info['fee']
        }



# Usage
data_loader = DataLoader('tseamcet.csv')
college_dataset = data_loader.load_data()
data_loader.display_info()
missing_values = data_loader.check_missing_values()
print('missing values')
print(missing_values)

# Encode the categorical variables and process the data
preprocessor = DataPreprocessor(college_dataset)
encoded_data = preprocessor.encode_features()


# Separate features and labels, split the data
X, Y = preprocessor.separate_features_labels('college_code', ['rank', 'gender', 'caste', 'branch_code'])
X_train, X_test, Y_train, Y_test = preprocessor.split_data()

# Train and evaluate the model
trainer = ModelTrainer()
trainer.train_model(X_train, Y_train)
accuracy = trainer.evaluate_model(X_test, Y_test)
print('Accuracy:', accuracy)

# Save the model to a file
dump(trainer.model, 'model.joblib')

# Predict using the trained model
predictor = Predictor(trainer.model, X_train.columns, college_dataset)

# Example input data
user_input = {
    'rank': 71654,
    'gender': 'F',
    'caste': 'BC_B',
    'branch_code': 'PHD',
}

# Encode the input data
encoded_input = [
    user_input['rank'],
    0 if user_input['gender'] == 'F' else 1,
    {'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8}[user_input['caste']],
    BRANCH[user_input['branch_code']][0]
]

# Assuming Predictor is already initialized as `predictor`
prediction_result = predictor.predict_college(encoded_input)

print('Predicted College Code:', prediction_result['college_code'])
print('College:', prediction_result['college'])
print('Students per Class:', prediction_result['students_per_class'])
print('Fee:', prediction_result['fee'])