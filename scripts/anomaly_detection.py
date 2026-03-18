import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

class AnomalyDetection:
    def __init__(self):
        self.iforest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)

    def train_iforest(self, X):
        self.iforest.fit(X)
        scores = self.iforest.decision_function(X)
        joblib.dump(self.iforest, 'models/iforest_model.pkl')
        return scores

    def detect_iforest(self, X):
        preds = self.iforest.predict(X)
        return preds  # -1 for anomaly, 1 for normal

    def train_dbscan(self, X):
        self.dbscan.fit(X)
        labels = self.dbscan.labels_
        return labels  # -1 for noise/outlier

    def detect_dbscan(self, X):
        labels = self.dbscan.fit_predict(X)
        return labels

    def evaluate_model(self, preds, true_labels):
        preds_binary = (preds == -1).astype(int)
        true_binary = true_labels.astype(int)
        precision = precision_score(true_binary, preds_binary, zero_division=0)
        recall = recall_score(true_binary, preds_binary, zero_division=0)
        f1 = f1_score(true_binary, preds_binary, zero_division=0)
        return {'precision': precision, 'recall': recall, 'f1': f1}