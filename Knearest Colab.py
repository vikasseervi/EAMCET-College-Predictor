# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gXYvpz9YQSYNyO5KCr5N33kNUtj7s-oS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score

# Load dataset
df = pd.read_csv("tseamcet.csv")

# Display basic information about the dataset
print(df.tail())
print(df.head())
print(df.info())
print(df.describe())
print("Shape of DataFrame:", df.shape)

# Display unique values in specific columns
print("Columns:", df.columns)
print("Unique Castes:", df["caste"].unique())
print("Unique Colleges:", df["college"].unique())
print("Unique Branches:", df["branch_code"].unique())

# Visualize missing values in the dataset
sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis', alpha=0.5)
plt.title('Missing Values Heatmap')
plt.show()

# Bar plot of college counts
plt.figure(figsize=(10, 5))
df['college_code'].value_counts().plot(kind='bar', color="#5cd65c", alpha=0.5)
plt.title('Number of Students per College')
plt.xlabel('College Code')
plt.ylabel('Number of Students')
plt.show()

# Caste distribution plot
max_caste = df['caste'].value_counts()
plt.figure(figsize=(15, 8))
plt.plot(max_caste, color="#2e2eb8", alpha=0.5)
plt.title('Caste Distribution')
plt.xlabel('Caste')
plt.ylabel('Count')
plt.show()

# Branch distribution for CSE students
category_true = df.loc[df['branch_code'] == 'CSE', 'caste'].value_counts()
plt.figure(figsize=(15, 8))
category_true.plot(kind='bar', color="#000066", alpha=0.5)
plt.title('Caste Distribution for CSE Branch')
plt.xlabel('Caste')
plt.ylabel('Count')
plt.xticks(rotation=360)
plt.legend()
plt.show()

# College and branch code count plot
plt.figure(figsize=(10, 5))
year_club = df.groupby(['college_code', 'branch_code']).size().head(300).plot(kind='bar', color="#ff6600", alpha=0.5)
plt.title('Branch Count per College Code')
plt.xlabel('College Code and Branch Code')
plt.ylabel('Count')
plt.legend()
plt.show()

# Prepare data for model training
eval_College = df.copy()  # Use a copy to avoid modifying original DataFrame

# Encode categorical variables
le = LabelEncoder()
le_caste = LabelEncoder()
eval_College['caste'] = le_caste.fit_transform(eval_College['caste'])  # Encode caste
eval_College['gender'] = eval_College['gender'].map({'F': 0, 'M': 1})
eval_College['college'] = le.fit_transform(eval_College['college'])
eval_College['branch_code'] = le.fit_transform(eval_College['branch_code'])

# Check the classes learned by the encoder
print("Classes:", le_caste.classes_)

# eval_College['caste'] = le.fit_transform(eval_College['caste'])
# le = LabelEncoder()
# le.fit(df['caste'])
#
# # Check the classes learned by the encoder
# print("Classes:", le.classes_)
#
# # Prepare new data
# new_data = {
#     'rank': 5000,
#     'gender': 'M',
#     'caste': 'OC',  # This is an unseen category causing KeyError
#     'branch_code': 'CSE'
# }
# Handle unseen caste values
# if new_data['caste'] not in le.classes_:
#     print(f"Unseen caste '{new_data['caste']}' encountered. Mapping to '<unknown>'.")
#     new_data['caste'] = '<unknown>'  # or some other placeholder
#
# # Transform caste with safety check for unseen values
# new_data['caste'] = le.transform([new_data['caste']])[0]



# Define features and target variable
X = eval_College[['rank', 'gender', 'caste', 'branch_code']]
y = eval_College['college']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Train the KNN model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predictions and metrics evaluation
y_pred = model.predict(X_test)

# Display Mean Absolute Error correctly
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Generate classification report with zero_division set to 0 to avoid warnings
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Calculate and display accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Example input values for prediction
new_data = {
    'rank': 5000,
    'gender': 'M',
    'caste': 'OC',
    'branch_code': 'CSE'
}

# Convert categorical values to match training data encoding
new_data['gender'] = 1 if new_data['gender'] == 'M' else 0
new_data['caste'] = le_caste.transform([new_data['caste']])[0]
new_data['branch_code'] = le.transform([new_data['branch_code']])[0]


# Prepare input features for prediction
input_features = np.array([[new_data['rank'], new_data['gender'], new_data['caste'], new_data['branch_code']]])

# Predict the college code and decode it back to original form if necessary
predicted_college_code = model.predict(input_features)
predicted_college = le.inverse_transform(predicted_college_code)[0]

print("Predicted College:", predicted_college)