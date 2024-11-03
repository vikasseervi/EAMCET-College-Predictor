# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#
#
# # Load the dataset
# url = 'tseamcet.csv'  # Update this to the actual path
# data = pd.read_csv(url)
#
# # Display the first few rows
# print(data.head())
#
# # Step 4: Data preprocessing
# # Check for missing values
# print(data.isnull().sum())
#
# # Handle missing values (you can modify this according to your analysis)
# data.ffill(inplace=True)
#
# # Feature selection
# features = data[['branch_code', 'college_code', 'rank']]  # Update based on actual feature names
# target = data['college']  # Update based on your target column
#
# # Encode categorical variables using .loc
# label_encoder = LabelEncoder()
# features.loc[:, 'branch_code'] = label_encoder.fit_transform(features['branch_code'])
# features.loc[:, 'college_code'] = label_encoder.fit_transform(features['college_code'])
# target = label_encoder.fit_transform(target)
#
#
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
# # Train a Random Forest Classifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Predictions
# y_pred = model.predict(X_test)
#
# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')
#
# # Classification Report
# print(classification_report(y_test, y_pred))
#
# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
#
# def predict_college(branch_code, college_code, rank):
#     input_data = pd.DataFrame([[branch_code, college_code, rank]], columns=['branch_code', 'college_code', 'rank'])
#     input_data['branch_code'] = label_encoder.transform(input_data['branch_code'])
#     input_data['college_code'] = label_encoder.transform(input_data['college_code'])
#     prediction = model.predict(input_data)
#     return label_encoder.inverse_transform(prediction)
#
# # Example prediction
# print(predict_college('example_branch_code', 'example_college_code', 10000))  # Update with real values


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('tseamcet.csv')
df.tail()