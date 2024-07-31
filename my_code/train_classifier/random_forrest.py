import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import copy
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import scipy.stats
from pdb import set_trace

def haar_wavelet_transform(x):
    #haar transform requires data to have length 64
    data = copy.deepcopy(x)

    assert len(data)<= 64, f"{len(data)} should not exceed 64"
    data = np.pad( data, (0, 64-len(data)), mode='constant', constant_values=1)
    
    result = []
   
    while len(data) > 1:
        # Compute the averages and differences
        averages = (data[::2] + data[1::2]) / 2
        differences = (data[::2] - data[1::2]) / 2
        result.append(differences)
        data = averages
    
    # Append the final averages to the result
    result.append(data)
    
    return result[::-1]

def time_bin(x, num_bins):
    bin_length = len(x)//num_bins
    bins = [x[i*bin_length:(i+1)*bin_length] for i in range(0, num_bins)]
    return bins


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    var = np.nanvar(list_values)
    rms = np.sqrt(np.nanmean(list_values**2))
    return [n5, n25, n75, n95, median, mean, var, rms]
 
def calculate_crossings(list_values):
    values_array = np.array(list_values)
    zero_crossing_indices = np.nonzero(np.diff(values_array > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(values_array > np.nanmean(values_array)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]
 
def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

def process_rf_data(X, y, knn, is_train=True):
    proc_X = []
    for idx, data in enumerate(X):
        if is_train:
            knn_counts = knn.predict_proba(np.expand_dims(data[-32:], 0)) * knn.get_params()["n_neighbors"]
            knn_counts[0, y[idx]] -= 1 #during training remove self to reduce bias
            knn_prob = knn_counts / 12
        else:
            knn_prob = knn.predict_proba(np.expand_dims(data[-32:], 0))

        wavelet_convolutions = haar_wavelet_transform(data)
        inputs = wavelet_convolutions + time_bin(data, 3) + time_bin(data, 1)
        
        features = [get_features(input) for input in inputs] + [[knn_prob[0][0]]]
        flattened_features = np.concatenate(features).ravel()
        proc_X.append(flattened_features)
    
    proc_X = np.array(proc_X)
    
    return proc_X, y