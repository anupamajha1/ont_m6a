"""
m6a_xg.py
Author: Sydney Kuhl <smkuhl@uw.edu>
This module predicts whether an
adenine is methylated or not. The
model is trained with ONT read sequence, 
read quality score and methylation score
from oxford nanopore. The model is a
XGBoost.
"""

import torch
import json
import numpy as np
import _pickle as pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, log_loss
import matplotlib.pyplot as plt

verbose = False

class M6AXG(XGBClassifier):
    def __init__(
        self,
        n_estimators=2,
        max_depth=2,
        learning_rate=1,
        min_child_weight=1,
        colsample_bytree=1.0,
        objective='binary:logistic',
        eval_metric=["logloss", "error", "aucpr"]
    ):
        """
        Constructor for the M6AXG, an XGBoost
        model for m6A calling.
        """
        super(M6AXG, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            colsample_bytree=colsample_bytree,
            objective=objective,
            eval_metric=eval_metric
        )
    
    def fit_supervised(
            self,
            training_data,
            X_valid=None,
            y_valid=None,
            verbose=1,
            device="cpu",
            tree_dump_path="",
            final_save_model="",
    ):
        """
        Training procedure for the supervised version
        of m6A XGBoost.
        :param training_data: tuple, training features, labels
        :param X_valid: numpy array, validation features
        :param y_valid: numpy array, validation labels
        :param verbose: int, at what boosting stage eval metrics
                            are printed to stdout
        :param device: str, GPU versus CPU, defaults to CPU
        :param tree_dump_path: str, path to save best model
        :param final_save_model: str, path to save final model
        :return: None
        """
    
        if device != 'cpu':
            # we need to send validation data to gpu
            x_tensor_on_cpu = torch.from_numpy(X_valid)
            y_tensor_on_cpu = torch.from_numpy(y_valid)
            x_tensor_on_gpu = tensor_on_cpu.to(device)
            y_tensor_on_gpu = tensor_on_cpu.to(device)
            
        X, y_train = training_data
        
        # flatten data into 2d array
        # this will have to change if input size is not 6x15
        X_val = X_valid.reshape(-1, 90) 
        X_train = X.reshape(-1, 90)
        print(f"Training features shape {X_train.shape}")
        print(f"Training labels shape {X_val.shape}")
        
        # evaluate on both the training and validation dataset
        evals = [(X_train, y_train), (X_val, y_valid)]

        # train the model, evaluating both on the training data and the validation data
        self.fit(X_train, y_train, eval_set=evals, verbose=True)
        
        results = self.evals_result()
        print(results.keys())
        print(results['validation_0'].keys())
        train_logloss = results["validation_0"]["logloss"]
        val_logloss = results["validation_1"]["logloss"]
        train_error = results["validation_0"]["error"]
        val_error = results["validation_1"]["error"]
        train_aucpr = results["validation_0"]["aucpr"]
        val_aucpr = results["validation_1"]["aucpr"]
        
        plt.plot(range(len(train_logloss)), train_logloss, label="Training")
        plt.plot(range(len(val_logloss)), val_logloss, label="Validation")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('XGBoost Log Loss')
        plt.legend()
        plt.savefig('xg_loss_chart.png')
        plt.clf()
            
        plt.plot(range(len(train_error)), train_error, label="Training")
        plt.plot(range(len(val_error)), val_error, label="Validation")
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('XGBoost Error')
        plt.legend()
        plt.savefig('xg_error_chart.png')
        plt.clf()
        
        plt.plot(range(len(train_aucpr)), train_aucpr, label="Training")
        plt.plot(range(len(val_aucpr)), val_aucpr, label="Validation")
        plt.xlabel('Iteration')
        plt.ylabel('AUPCR')
        plt.title('XGBoost AUCPR')
        plt.legend()
        plt.savefig('xg_aucpr_chart.png')
        plt.clf()

        # save model
        self.save_model(final_save_model)
        
        # save the tree dump to a file for easier interpretability
        tree_dump = self.get_booster().get_dump()
        
        with open("final_xgboost_classifier_trees.txt", "w") as tree_file:
            for i, tree in enumerate(tree_dump):
                tree_file.write(f"Tree {i}:\n")
                tree_file.write(tree)
                tree_file.write("\n\n")
                
        # also save the results
        with open('xg_results.json', "w") as json_file:
            json.dump(results, json_file)
            
        print("done!")
    
