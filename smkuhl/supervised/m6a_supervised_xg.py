#!/usr/bin/env python3
"""
m6a_supervised_cnn.py
Author: Sydney Kuhl <smkuhl@uw.edu>
This module predicts whether an
adenine is methylated or not. The
model is trained with ONT read sequence, 
read quality score and methylation score
from oxford nanopore. The model is a
XGBoost.
"""

import torch
import argparse
import numpy as np
import configparser
import _pickle as pickle
from torchsummary import summary
from smkuhl.supervised.m6a_xg import M6AXG

def m6ALoader(
    train_path,
    val_path,
    input_size
):
    """
    This generator returns a training
    data generator as well as validation
    features and labels.
    :param train_path: str, path where
                            the training
                            data is stored.
    :param input_size: int, number of input
                            channels
    :param val_path: str, path where the
                          val data is stored.
    :return: training_data: tuple (X_train, y_train),
             X_val: validation features,
             y_val: validation labels
    """
    # Load training data
    train_data = np.load(train_path, allow_pickle=True)
    print(list(train_data.keys()))

    # Load training and validation
    # features and labels. Sometimes
    # we want to train on input subsets,
    # this will achieve that.
    
    X_train = train_data["features"]
    X_train = X_train[:, 0:input_size, :]
    y_train = train_data["labels"]
    print(f"y_train: {y_train.shape}, {y_train}, {np.unique(y_train)}")

    # Load validation data
    val_data = np.load(val_path, allow_pickle=True)
    X_val = val_data["features"]
    X_val = X_val[:, 0:input_size, :]
    y_val = val_data["labels"]

    print(
        f"Training features shape {X_train.shape},"
        f" training labels shape: {y_train.shape}"
    )
    print(
        f"Validation features shape {X_val.shape}, "
        f" validation labels shape: {y_val.shape}"
    )
    
    return (X_train, y_train), X_val, y_val


def run(config_file, train_chem):
    """
    Run data preprocess and model training.
    :param config_file: str, path to config
                            file.
    :param train_chem: str, which chemistry
                            to train.
    :return: None
    """
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(config_file)
    # get parameters for the
    # relevant chemistry
    rel_config = config[train_chem]
    # Number of input channels
    input_size = int(rel_config["input_size"])
    # length of input sequence
    input_length = int(rel_config["input_length"])
    # path to training data set
    train_data = rel_config["sup_train_data"]
    # path to validation data set
    val_data = rel_config["sup_val_data"]
    # cpu or cuda for training
    device = rel_config["device"]
    # path to save best model
    best_save_model = rel_config["best_supervised_model_name"]
    # path to save final model
    final_save_model = rel_config["final_supervised_model_name"]
    # learning rate
    sup_lr = float(rel_config["sup_lr"])

    model = M6AXG(
                n_estimators=32,
                max_depth=2,
                learning_rate=1,
                objective='binary:logistic'
            )
    
    # Print model architecture summary
    # summary_str = summary(model, input_size=(input_size, input_length))

    # Get training data generator
    # and validation data.
    train, X_val, y_val = m6ALoader(
        train_data,
        val_data,
        input_size
    )

    # Train the model
    model.fit_supervised(
        train,
        X_valid=X_val,
        y_valid=y_val,
        verbose=1,
        device=device,
        tree_dump_path="XGBoost_Tree.txt", 
        final_save_model="final_xgboost_model.json", # final_save_model
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="config.yml", help="path to the config file."
    )

    parser.add_argument(
        "--train_chem",
        type=str,
        default="train_ONT_chemistry",
        help="Which chemistry to validate. The name should match the section header in the config file.",
    )

    args = parser.parse_args()

    print(f"Training a {args.train_chem} " f"supervised XGBoost model.")

    run(args.config_file, args.train_chem)


if __name__ == "__main__":
    main()
