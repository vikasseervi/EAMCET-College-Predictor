# main.py
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from data_visualizer import DataVisualizer
from model_trainer import ModelTrainer
from predictor import Predictor
from util import BRANCH

# Load data
# data_loader = DataLoader('tseamcet.csv')
# college_dataset = data_loader.load_data()
# data_loader.display_info()
# missing_values = data_loader.check_missing_values()
# print('missing values:', missing_values)

# Visualize the data
# visualizer = DataVisualizer()DataVisualizer
# visualizer.plot_category_distribution(college_dataset, 'branch')
# visualizer.plot_category_distribution(college_dataset, 'gender')
# visualizer.plot_feature_vs_target(college_dataset, 'rank', 'college_code')

# Encode features and preprocess data
# preprocessor = DataPreprocessor(college_dataset)
# encoded_data = preprocessor.encode_features()
# X, Y = preprocessor.separate_features_labels('college_code', ['rank', 'gender', 'caste', 'branch_code'])
# X_train, X_test, Y_train, Y_test = preprocessor.split_data()
# visualizer.plot_correlation_heatmap(encoded_data)

# Train model
trainer = ModelTrainer()
# trainer.train_model(X_train, Y_train)
# trainer.save_model('model_list_college.joblib')
# accuracy = trainer.evaluate_model(X_test, Y_test)
# print(f"Model Accuracy: {accuracy}")
trainer.load_model()

# Predict top colleges
predictor = Predictor(trainer.model, X_train.columns, college_dataset)

user_input = {'rank': 71654, 'gender': 'M', 'caste': 'BC_B', 'branch_code': 'CSE'}

encoded_input = [
    user_input['rank'],
    0 if user_input['gender'] == 'F' else 1,
    {'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8}[user_input['caste']],
    BRANCH[user_input['branch_code']][0]
]
top_colleges = predictor.predict_top_colleges(encoded_input, top_n=10)

# Display top colleges
for idx, college in enumerate(top_colleges, start=1):
    print(f"Top {idx} Predicted College Code: {college['college_code']}")
    print(f"College: {college['college']}")
    print(f"Students per Class: {college['students_per_class']}")
    print(f"Fee: {college['fee']}\n")
