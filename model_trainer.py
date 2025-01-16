# model_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load


class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train_model(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def evaluate_model(self, X_test, Y_test):
        X_test_prediction = self.model.predict(X_test)
        return accuracy_score(X_test_prediction, Y_test)

    def save_model(self, file_path):
        dump(self.model, file_path)

    def load_model(self):
        self.model = load('model_list_college.joblib')
