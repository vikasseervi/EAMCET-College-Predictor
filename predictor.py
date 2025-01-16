# predictor.py
import pandas as pd
import numpy as np

class Predictor:
    def __init__(self, model, feature_columns, data):
        self.model = model
        self.feature_columns = feature_columns
        self.data = data

    def predict_top_colleges(self, input_data, top_n=5):
        input_df = pd.DataFrame([input_data], columns=self.feature_columns)
        probabilities = self.model.predict_proba(input_df)[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_college_codes = self.model.classes_[top_indices]

        top_colleges = []
        for college_code in top_college_codes:
            college_info = self.data[self.data['college_code'] == college_code].iloc[0]
            top_colleges.append({
                'college_code': college_code,
                'college': college_info['college'],
                'students_per_class': college_info['students_per_class'],
                'fee': college_info['fee']
            })

        return top_colleges
