#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:13:26 2023

@author: morgan
"""
import numpy as np
import argparse

v_hash = np.vectorize(hash, doc='vectorized hash')

# =============================================================================
# # args for testing
# args = argparse.Namespace(positive_file='HG002_2_00.npz', negative_file='HG002_3_00.npz',
#                           cutoff_score=100, output_prefix='set_00', 
#                           split=['train', 1e6, 'validation', 2.5e5,  'test', 2.5e5]) 
# 
# =============================================================================

def filter_data(data_set, cutoff):
    mask = data_set['features'][:,5,7] >= cutoff
    data_filt = {key:value[mask] for (key,value) in data_set.items()}
    return(data_filt)

def random_subset(data_set, count):
    idx = np.arange(data_set['features'].shape[0])
    idx = np.random.choice(idx, int(count), replace=False)
    return( {key:value[idx] for (key,value) in data_set.items()})

def split_by_read(data_set, count_list):
    MOD_VAL = 1000
    if sum(count_list) > data_set['features'].shape[0]:
        raise RuntimeError('dataset is not large enough for given counts')
    scaled_counts = np.array(count_list) * MOD_VAL / np.sum(count_list)
    split_points = [0]
    for count in scaled_counts:
        split_points.append(split_points[-1] + count)
    read_id_hash = v_hash(data_set['read_ids']) % 1000
    out_list = []
    for i in range(len(count_list)):
        mask = (read_id_hash > split_points[i]) & (read_id_hash <= split_points[i+1])
        out_set = {key:value[mask] for (key,value) in data_set.items()}
        out_set = random_subset(out_set, count_list[i])
        out_list.append(out_set)
    return(out_list)
    
def shuffle_merge(data_set_list):
    merged = {key:np.concatenate([data_set[key] for data_set in data_set_list]) for key in data_set_list[0].keys()}
    return(random_subset(merged, merged['features'].shape[0]))

def main(args):
    positive_set = dict(np.load(args.positive_file))
    negative_set = dict(np.load(args.negative_file))
    
    positive_set = filter_data(positive_set, args.cutoff_score)
    # negative_set = filter_data(negative_set, args.cutoff_score)

    pos_split = split_by_read(positive_set, np.array(args.split_counts)/2)
    neg_split = split_by_read(negative_set, np.array(args.split_counts)/2)
    
    for i, suffix in enumerate(args.split_names):
        merged_data = shuffle_merge([pos_split[i], neg_split[i]])
        merged_data['features'] = np.array(merged_data['features'], dtype=np.float32)
        merged_data['features'][:,4:,:] = merged_data['features'][:,4:,:] / 255
        file_name = args.output_prefix + "_" + suffix + ".npz"
        np.savez(file_name, **merged_data)
        
    return(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='shuffle and split data into train and test sets')
    parser.add_argument('-p', '--positive_file', type=str, 
                        help='file path to positive labeled data') 
    parser.add_argument('-n', '--negative_file', type=str, 
                        help='file path to positive labeled data') 
    parser.add_argument('-c', '--cutoff_score', type=int, default=155, 
                        help='remove sites with score below cutoff from output data') 
    parser.add_argument('-o', '--output_prefix', type=str, default='output', 
                        help='output file name prefix') 
    parser.add_argument('-s', '--split', nargs='+', default=['train', 5e6, 'validation', 1e6,  'test', 1e6], 
                        help='output sufix and count. ie. -s train 5e6 val 1e6 test 1e6')
    args = parser.parse_args()
    args.split_names = []
    args.split_counts = []
    for i in range(len(args.split)):
        if i % 2 == 0:   
            args.split_names.append(args.split[i])
        else:
            args.split_counts.append(int(args.split[i]))
    main(args)
    # print(args.split_names)
    # print(args.split_counts)
