from joblib import load

# Load the model
loaded_model = load('model.joblib')

# Initialize the Predictor with the loaded model
predictor = Predictor(loaded_model, X_train.columns, college_dataset)

# Example input data
user_input = {
    'rank': 71654,
    'gender': 'F',
    'caste': 'BC_B',
    'branch_code': 'CSE',
}

# Encode the input data
encoded_input = [
    user_input['rank'],
    0 if user_input['gender'] == 'F' else 1,
    {'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8}[user_input['caste']],
    BRANCH[user_input['branch_code']][0]
]

# Predict the output using the encoded input
prediction_result = predictor.predict_college(encoded_input)

print('Predicted College Code:', prediction_result['college_code'])
print('College:', prediction_result['college'])
print('Students per Class:', prediction_result['students_per_class'])
print('Fee:', prediction_result['fee'])
