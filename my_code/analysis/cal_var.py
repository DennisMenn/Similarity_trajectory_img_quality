import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from pdb import set_trace

def process_data(data_path):
    df = pd.read_csv(data_path)
    df_X = df.iloc[:, 3].values #next        
    df_Y = df.iloc[:, 2].values

    natural = []
    unnatural = []

    for idx, element in enumerate(df_X):
        if df_Y[idx] == 'natural':
            natural.append(ast.literal_eval(df_X[idx]))
        elif df_Y[idx] == "unnatural":
            unnatural.append(ast.literal_eval(df_X[idx]))
    natural = np.array(natural)
    unnatural = np.array(unnatural)

    return natural, unnatural

if __name__ == "__main__":
    data_path = "/home/dym349/Desktop/diffusion_models/Image_quality/my_code/create_dataset/dataset/balance/dataset.csv"
    save_path = "std.png"
    nat, unnat = process_data(data_path)
    nat_std = nat.std(axis=0)
    unnat_std = unnat.std(axis=0)
    
    dim = (10,37)
    x_axis = list(range(5,50))

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, nat_std[dim], label='Natural')
    plt.plot(x_axis, unnat_std[dim], label='Unnatural')
    plt.legend(fontsize=12)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Standard Deviation", fontsize=14)
    plt.title("Standard Deviation of Similarity Trajectory", fontsize=16)
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)

