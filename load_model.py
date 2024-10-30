from joblib import load

from util import BRANCH
import pandas as pd

class CollegePredictor:
    def __init__(self, data):
        # Load the model only once when the class is initialized
        self.model = load('model.joblib')
        self.feature_columns = ['rank', 'gender', 'caste', 'branch_code']
        self.data = data

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

# Instantiate the predictor once
college_predictor = CollegePredictor(pd.read_csv('tseamcet.csv'))

# Example input data
user_input = {
    'rank': 67000,
    'gender': 'M',
    'caste': "ST",
    'branch_code': 'CSE',
}

# Encode the input data
encoded_input = [
    user_input['rank'],
    0 if user_input['gender'] == 'F' else 1,
    {'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8}[user_input['caste']],
    BRANCH[user_input['branch_code']][0]
]

# Get prediction
result = college_predictor.predict_college(encoded_input)

# Display result
print('Predicted College Code:', result['college_code'])
print('College:', result['college'])
print('Students per Class:', result['students_per_class'])
print('Fee:', result['fee'])



""" New code by GPT """


# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split, GridSearchCV
# import pandas as pd
#
#
# class DataLoader:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.college_dataset = None
#
#     def load_data(self):
#         self.college_dataset = pd.read_csv(self.file_path)
#         return self.college_dataset
#
#     def display_info(self):
#         print(self.college_dataset.shape)
#         print(self.college_dataset.head())
#         print(self.college_dataset.describe())
#
#     def check_missing_values(self):
#         return self.college_dataset.isnull().sum()
#
#
# class DataPreprocessor:
#     def __init__(self, data):
#         self.data = data
#         self.X = None
#         self.Y = None
#         self.scaler = StandardScaler()
#
#     def encode_features(self):
#         # Map categorical variables to numeric values
#         self.data['gender'] = self.data['gender'].map({'F': 0, 'M': 1})
#         self.data['caste'] = self.data['caste'].map({
#             'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8
#         })
#         # Assign arbitrary but consistent numeric codes to `branch_code`
#         branch_code_mapping = {code: idx for idx, code in enumerate(self.data['branch_code'].unique())}
#         self.data['branch_code'] = self.data['branch_code'].map(branch_code_mapping)
#
#         return self.data
#
#     def separate_features_labels(self, target_column, selected_features=None):
#         if selected_features:
#             self.X = self.data[selected_features]
#         else:
#             self.X = self.data.drop(target_column, axis=1)
#         self.Y = self.data[target_column]
#         return self.X, self.Y
#
#     def split_data(self, test_size=0.2, random_state=3):
#         # Scale the features
#         self.X = self.scaler.fit_transform(self.X)
#         return train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)
#
#
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
#
# class ModelTrainer:
#     def __init__(self):
#         # Setting parameters to improve the model performance
#         self.model = RandomForestClassifier(class_weight='balanced', random_state=3)
#
#     def train_model(self, X_train, Y_train):
#         # Hyperparameter tuning with stratified cross-validation
#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [10, 20, 30],
#             'min_samples_split': [2, 5, 10]
#         }
#         # Using StratifiedKFold to ensure balanced splits across classes
#         skf = StratifiedKFold(n_splits=3)
#         grid_search = GridSearchCV(self.model, param_grid, cv=skf, scoring='accuracy')
#         grid_search.fit(X_train, Y_train)
#         self.model = grid_search.best_estimator_
#         print("Best parameters found: ", grid_search.best_params_)
#
#
# # Load and preprocess data
# data_loader = DataLoader('tseamcet.csv')
# college_dataset = data_loader.load_data()
# data_loader.display_info()
# missing_values = data_loader.check_missing_values()
# print('missing values:', missing_values)
#
# # Encode the categorical variables and process the data
# preprocessor = DataPreprocessor(college_dataset)
# encoded_data = preprocessor.encode_features()
#
# # Separate features and labels, split the data
# X, Y = preprocessor.separate_features_labels('college_code', ['rank', 'gender', 'caste', 'branch_code'])
# X_train, X_test, Y_train, Y_test = preprocessor.split_data()
#
# # Train and evaluate the model
# trainer = ModelTrainer()
# trainer.train_model(X_train, Y_train)
# accuracy = trainer.evaluate_model(X_test, Y_test)
# print('Improved Accuracy:', accuracy)
