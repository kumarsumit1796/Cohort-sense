import autokeras
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import os

import warnings

warnings.filterwarnings("ignore")


class AutoKerasModule:

    # Function returns trained model by Tabular classifier of Auto-Keras
    @staticmethod
    def build_model(train_test, path):
        clf = autokeras.TabularClassifier(path, verbose=True)
        clf.fit(train_test[0], train_test[2])
        # retrain the best model
        clf.final_fit(train_test[0], train_test[2], train_test[1], train_test[3], retrain=True)
        return clf

    # Function that returns predicted values on test data
    @staticmethod
    def predict_and_evaluate(train_test, path):
        model = AutoKerasModule.build_model(train_test, path)
        predict = model.predict(train_test[1])
        accuracy = model.evaluate(train_test[1], train_test[3])
        return predict, accuracy

    @staticmethod
    def eval_metrics(y_test, predict):
        print("Accuracy:", accuracy_score(y_test, predict))
        print("Precision:", precision_score(y_test, predict))
        print("Recall:", recall_score(y_test, predict))
        return accuracy_score(y_test, predict), precision_score(y_test, predict), recall_score(y_test, predict)
