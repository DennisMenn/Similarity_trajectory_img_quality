import pandas as pd
import numpy as np
import argparse
import ast
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

from torch.utils.data import DataLoader, TensorDataset
from joblib import dump

from random_forrest import process_rf_data

from pdb import set_trace

class Train():
    def __init__(self, data_path, k_folds, is_transform=True, dim=0, dump_dir=None, seed=0):
        self.data_path = data_path
        self.kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)        
        self.Y= []
        self.X= []
        self.accs = []
        self.dim = dim #truncate data dim
        self.feature_importance = []
        self.dump_dir = dump_dir
        self.seed = seed
        self.is_transform = is_transform

    def plot_feature_importance(self):
        window_size = 3
        self.feature_importance = np.array(self.feature_importance).mean(0)
        moving_average = np.convolve(self.feature_importance, np.ones(window_size) / window_size, mode='same')
        plt.figure(figsize=(12, 6))
        plt.title("Feature importances")
        plt.bar(range(len(self.feature_importance)), self.feature_importance,
            color="r", align="center")
        plt.plot(moving_average)
        plt.savefig("tree_importance.png")
        
    def process_data(self):
        df = pd.read_csv(self.data_path)
        df_X = df.iloc[:, 3].values #next        
        df_Y = df.iloc[:, 2].values

        for idx, element in enumerate(df_X):
            self.X.append(ast.literal_eval(element))
            self.Y.append(0 if df_Y[idx] == "unnatural" else 1)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def training_transform(self, model, X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier(n_neighbors=13, metric='cosine')
        knn.fit(X_train[:,-32:], y_train)

        X_train, y_train = process_rf_data(X_train, y_train, knn, True)
        X_test, y_test = process_rf_data(X_test, y_test, knn, False)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        feature_importances = model.feature_importances_
        self.feature_importance.append(feature_importances)

        return model, knn, accuracy
    
    def training_vanilla(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        feature_importances = model.feature_importances_
        self.feature_importance.append(feature_importances)

        return model, accuracy

    def k_fold_training(self):
        for fold, (train_ids, test_ids) in enumerate(self.kf.split(self.X)):
            # Split data
            X, X_test = self.X[train_ids][:,self.dim[0]:self.dim[1]], self.X[test_ids][:,self.dim[0]:self.dim[1]]
            y, y_test = self.Y[train_ids], self.Y[test_ids]

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

            model = RandomForestClassifier(n_estimators=1000, random_state=0)

            if self.is_transform:
                model, knn, test_accuracy = self.training_transform(model, X, X_test, y, y_test)
            else:
                model, test_accuracy = self.training_vanilla(model, X, X_test, y, y_test)  

            # print(f'Fold {fold}, Accuracy: {test_accuracy*100}%')
            self.accs.append(test_accuracy)
        
        acc_mean = np.mean(self.accs)
        acc_sem = stats.sem(self.accs)
        print(f'Average K-Fold Accuracy: {acc_mean*100:.2f}% (SEM: {acc_sem*100:.2f}%)')
        
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)
            dump(model, f'{self.dump_dir}/rf.joblib')
            dump(scaler, f'{self.dump_dir}/scaler.joblib')
            if self.is_transform: dump(knn, f'{self.dump_dir}/knn.joblib')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to demonstrate argparse with optional arguments.")
    parser.add_argument('--seed', type=int, default=0, help='seed to use')
    parser.add_argument('--dim', type=int, default=0, help='initial dim to truncate')

    data_path = "/home/dym349/Desktop/diffusion_models/Image_quality/my_code/create_dataset/dataset/balance/dataset.csv"
    dum_dir = "models/balanced550-10-folds"

    args = parser.parse_args()
    seed = args.seed
    dim = (9,38)
    is_transform = False

    np.random.seed(seed)
    trainer = Train(data_path, k_folds=10, dim=dim, seed=seed, is_transform=is_transform)
    trainer.process_data()
    trainer.k_fold_training()
    trainer.plot_feature_importance()