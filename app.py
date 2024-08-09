import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


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
        plt.show()


class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.Y = None

    def encode_features(self):
        # Encode gender, caste, branch_code, branch, and college as categorical variables
        self.data['gender'] = self.data['gender'].map({'F': 0, 'M': 1})
        self.data['caste'] = self.data['caste'].map(
            {'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8})
        self.data = pd.get_dummies(self.data, columns=['branch_code', 'branch'], drop_first=True)

        # Label encode the college_code and college
        label_encoder = LabelEncoder()
        self.data['college_code'] = label_encoder.fit_transform(self.data['college_code'])
        self.data['college'] = label_encoder.fit_transform(self.data['college'])

        return self.data

    def separate_features_labels(self, target_column):
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


class Predictor:
    def __init__(self, model, columns, original_data):
        self.model = model
        self.columns = columns
        self.original_data = original_data

    def predict(self, input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        input_df = pd.DataFrame(input_data_reshaped, columns=self.columns)
        prediction = self.model.predict(input_df)
        return prediction[0]

    def get_college_details(self, college_code):
        college_row = self.original_data[self.original_data['college_code'] == college_code]
        if not college_row.empty:
            return college_row[['college_code', 'college', 'students_per_class', 'fee']].iloc[0]
        else:
            return None


# Usage
data_loader = DataLoader('tseamcet.csv')
college_dataset = data_loader.load_data()
data_loader.display_info()
missing_values = data_loader.check_missing_values()
print(missing_values)

# Encode the categorical variables and process the data
preprocessor = DataPreprocessor(college_dataset)
encoded_data = preprocessor.encode_features()

# Plotting
visualizer = DataVisualizer()
visualizer.plot_category_distribution(college_dataset, 'branch')
visualizer.plot_category_distribution(college_dataset, 'gender')
visualizer.plot_feature_vs_target(college_dataset, 'rank', 'college_code')
visualizer.plot_correlation_heatmap(encoded_data)

# Separate features and labels, split the data
X, Y = preprocessor.separate_features_labels('college_code')
X_train, X_test, Y_train, Y_test = preprocessor.split_data()

# Train and evaluate the model
trainer = ModelTrainer()
trainer.train_model(X_train, Y_train)
accuracy = trainer.evaluate_model(X_test, Y_test)
print('Accuracy:', accuracy)

# Predict using the trained model
predictor = Predictor(trainer.model, X_train.columns, college_dataset)

# Example input data
user_input = {
    'rank': 71654,
    'gender': 'F',
    'caste': 'BC_B',
    'branch_code': 'CSE',
    'branch': 'COMPUTER SCIENCE AND ENGINEERING'  # Example branch
}

# Encode the input data
encoded_input = [
    user_input['rank'],
    0 if user_input['gender'] == 'F' else 1,
    {'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8}[user_input['caste']]
]

# Add one-hot encoded branch_code and branch
branch_codes = list(college_dataset['branch_code'].unique())
branches = list(college_dataset['branch'].unique())

for branch_code in branch_codes[1:]:  # drop_first=True
    encoded_input.append(1 if user_input['branch_code'] == branch_code else 0)

for branch in branches[1:]:  # drop_first=True
    encoded_input.append(1 if user_input['branch'] == branch else 0)

# Ensure the encoded input matches the feature set
while len(encoded_input) < len(X_train.columns):
    encoded_input.append(0)

college_prediction = predictor.predict(encoded_input)
college_details = predictor.get_college_details(college_prediction)

if college_details is not None:
    print('Predicted College Code:', college_details['college_code'])
    print('Predicted College Name:', college_details['college'])
    print('Students Per Class:', college_details['students_per_class'])
    print('Fee:', college_details['fee'])
else:
    print('College details not found for the predicted college code.')
