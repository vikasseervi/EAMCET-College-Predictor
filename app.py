from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from data_visualizer import DataVisualizer
from model_trainer import ModelTrainer
from predictor import Predictor
from util import BRANCH

# Initialize Flask app
app = Flask(__name__)

# Initialize model
trainer = ModelTrainer()
trainer.load_model()
feature_columns = ['rank', 'gender', 'caste', 'branch_code']
college_dataset = pd.read_csv('tseamcet.csv')
predictor = Predictor(trainer.model, feature_columns, college_dataset)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/results', methods=['POST'])
def results():
    # Get form data
    user_input = {
        'rank': int(request.form['rank']),
        'gender': request.form['gender'],
        'caste': request.form['caste'],
        'branch_code': request.form['branch_code'].split(' -')[0]
    }

    # Encode input
    encoded_input = [
        user_input['rank'],
        0 if user_input['gender'] == 'F' else 1,
        {'SC': 1, 'ST': 2, 'BC_A': 3, 'BC_B': 4, 'BC_C': 5, 'BC_D': 6, 'BC_E': 7, 'OC': 8}[user_input['caste']],
        BRANCH[user_input['branch_code']][0]
    ]

    # Predict top colleges
    top_colleges = predictor.predict_top_colleges(encoded_input, top_n=10)

    return render_template('results.html', top_colleges=top_colleges)

@app.route('/api/options', methods=['GET'])
def get_options():

    branches = []
    for branch_code, (id, branch_name) in BRANCH.items():
        branches.append([f"{branch_code} - {branch_name}"]);

    options = {
        "castes": ["SC", "ST", "BC_A", "BC_B", "BC_C", "BC_D", "BC_E", "OC"],
        "branches": branches
    }
    return jsonify(options)

if __name__ == '__main__':
    app.run(debug=True)