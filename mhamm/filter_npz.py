#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:03:09 2023

@author: morgan
"""
import numpy as np
import argparse


def filter_data(data_set, cutoff, label):
    score_mask = (data_set['features'][:,5,7] >= cutoff)
    if label is not None:
        label_mask = data_set['labels'] == label
    else:
        label_mask = np.ones(data_set['labels'].shape)
    mask = score_mask | (~label_mask)
    data_filt = {key:value[mask] for (key,value) in data_set.items()}
    return(data_filt)

def main(args):
    data = np.load(args.npz_file)
    data_filt = filter_data(data, args.cutoff_score, [args.label])
    np.savez(args.output_path, **data_filt)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter a npz file, removing entries with center base score below a cutoff')
    parser.add_argument('npz_file', help='the npz file to filter')
    parser.add_argument('-l', '--label', type=int, default=None,
                        help='only filter entries with this label') 
    parser.add_argument('-c', '--cutoff_score', type=int, default=155, 
                        help='remove sites with score below cutoff from output data') 
    parser.add_argument('-o', '--output_path', type=str, default='output.npz', 
                        help='output file name') 
    args = parser.parse_args()
    
    main(args)


