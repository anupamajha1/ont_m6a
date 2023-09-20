import torch
import argparse
import numpy as np
import configparser
import pandas as pd
import _pickle as pickle
from m6a_cnn import M6ANet
from m6a_semi_supervised_cnn import tdc, count_pos_neg, make_one_hot_encoded


def make_ont_predictions(best_sup_save_model, data_npz, output_obj, device="cuda"):
    # Load the supervised model for transfer learning
    model = M6ANet()
    with open(best_sup_save_model, "rb") as fp:
        model.load_state_dict(pickle.load(fp))
        
    model = model.to(device)
    
    val_data = np.load(data_npz)
    
    X_val = np.array(val_data['features'], dtype=float)
    print(f"X_val: {X_val.shape}")
    dorado_score = X_val[:, 5, 7]/255.0
    X_val[:, 4, :] = X_val[:, 4, :]/255.0
    X_val[:, 5, :] = X_val[:, 5, :]/255.0
    y_val = np.array(val_data['labels'], dtype=int)
    read_ids = val_data['read_ids']
    print("read_ids: ", len(np.unique(read_ids)))
    positions = val_data['positions']
    # convert to one hot encoded
    y_val_ohe = make_one_hot_encoded(y_val)

    # convert data to tensors
    X_val = torch.tensor(X_val).float()
    y_val_ohe = torch.tensor(y_val_ohe).float()
    X_val = X_val.to(device)
    y_val_ohe = y_val_ohe.to(device)
    
    preds_y = model.predict(X_val, device=device)
    total_len = len(preds_y)
    
    read_ids = read_ids[0:total_len][:, np.newaxis]
    positions = positions[0:total_len][:, np.newaxis]
    y_val = y_val[0:total_len][:, np.newaxis]
    preds_y = preds_y[0:total_len, 1].numpy()[:, np.newaxis]
    dorado_score = dorado_score[0:total_len][:, np.newaxis]
    
    print(f"read_ids: {read_ids.shape}")
    print(f"positions: {positions.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"preds_y: {preds_y.shape}")
    print(f"dorado_score: {dorado_score.shape}")
    
    output_arr = np.concatenate((read_ids, positions, y_val, dorado_score, preds_y), axis=1)
    np.savez(output_obj, preds=output_arr)
    

    
best_sup_save_model="../models/m6A_ONT_semi_supervised_cnn_5M_set2.best.torch.pickle"
data_npz="/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/all_sites_npz/merged_00_100p_20k.npz"
output_obj="../results/merged_00_100p_20k_autocorr_input_5M_set2.npz"
make_ont_predictions(best_sup_save_model, data_npz, output_obj)

best_sup_save_model="../models/m6A_ONT_semi_supervised_cnn_5M_set3.best.torch.pickle"
data_npz="/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/all_sites_npz/merged_00_100p_20k.npz"
output_obj="../results/merged_00_100p_20k_autocorr_input_5M_set3.npz"
make_ont_predictions(best_sup_save_model, data_npz, output_obj)