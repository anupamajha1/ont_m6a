#!/usr/bin/env python3
"""
m6a_ont_data.py
Author: Anupama Jha <anupamaj@uw.edu>
The script takes positive and negative
objects Morgan Hamm generated and makes
a ML object from it. 
"""

import argparse
import numpy as np
import configparser
from sklearn.model_selection import StratifiedKFold

def read_data(positive_path, negative_path, input_size=6):
    # Load positive_data
    positive_data = np.load(positive_path, allow_pickle=True)
    print(list(positive_data.keys()))
    
    X_pos = positive_data["features"]
    X_pos = np.array(X_pos[:, 0:input_size, :], dtype=float)
    X_pos[:, 4, :] = X_pos[:, 4, :]/255.0
    X_pos[:, 5, :] = X_pos[:, 5, :]/255.0
    y_pos = positive_data["labels"]
    read_pos = positive_data["read_ids"]
    print(f"X_pos: {X_pos.shape}, {X_pos[0]}")
    print(f"y_pos: {y_pos.shape}")
    
    X_pos = X_pos[0:4000000]
    y_pos = y_pos[0:4000000]
    read_pos = read_pos[0:4000000]
    
    # Load negative_data
    negative_data = np.load(negative_path, allow_pickle=True)
    print(list(negative_data.keys()))
    
    X_neg = negative_data["features"]
    X_neg = np.array(X_neg[:, 0:input_size, :], dtype=float)
    X_neg[:, 4, :] = X_neg[:, 4, :]/255.0
    X_neg[:, 5, :] = X_neg[:, 5, :]/255.0
    y_neg = negative_data["labels"]
    read_neg = negative_data["read_ids"]
    
    print(f"X_neg: {X_neg.shape}, {X_neg[0]}")
    print(f"y_neg: {y_neg.shape}")
    
    X_neg = X_neg[0:4000000]
    y_neg = y_neg[0:4000000]
    read_neg = read_neg[0:4000000]
    
    X_all = np.concatenate((X_pos, X_neg), axis=0)
    y_all = np.concatenate((y_pos, y_neg), axis=0)
    read_all = np.concatenate((read_pos, read_neg), axis=0)
    
    print(f"X_all: {X_all.shape}, {X_all[0]}")
    print(f"y_all: {y_all.shape}")
    print(f"read_all: {read_all.shape}")
    
    skf = StratifiedKFold(n_splits=5)
    X_train = None
    X_val = None
    X_test = None
    y_train = None
    y_val = None
    y_test = None
    read_train = None
    read_val = None
    read_test = None
    for i, (cv_index, test_index) in enumerate(skf.split(X_all, y_all)):
        print(i)
        X_cv = X_all[cv_index, :, :]
        y_cv = y_all[cv_index]
        read_cv = read_all[cv_index]
        X_test = X_all[test_index, :, :]
        y_test = y_all[test_index]
        read_test = read_all[test_index]
        read_test = read_all[test_index]
        for i, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
            print(i)
            X_train = X_cv[train_index, :, :]
            y_train = y_cv[train_index]
            read_train = read_cv[train_index]
            X_val = X_cv[val_index, :, :]
            y_val = y_cv[val_index]
            read_val = read_cv[val_index]
            break
        break
        
    print("Train: ", X_train.shape, y_train.shape, read_train.shape)
    print("Val: ", X_val.shape, y_val.shape, read_val.shape)
    print("Test: ", X_test.shape, y_test.shape, read_test.shape)
            
    np.savez("../data/HG002_2_3_00_train.npz", features=X_train, labels=y_train, read_ids=read_train)
    np.savez("../data/HG002_2_3_00_val.npz", features=X_val, labels=y_val, read_ids=read_val)
    np.savez("../data/HG002_2_3_00_test.npz", features=X_test, labels=y_test, read_ids=read_test)
        
    
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--positive_path", type=str, default="../data/HG002_2_00.npz", help="path to the positive data file."
    )
    
    parser.add_argument(
        "--negative_path", type=str, default="../data/HG002_3_00.npz", help="path to the negative data file."
    )

    args = parser.parse_args()

    read_data(args.positive_path, args.negative_path)


if __name__ == "__main__":
    main()