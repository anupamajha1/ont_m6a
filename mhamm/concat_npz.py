#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:59:58 2023

@author: morgan
"""
import numpy as np
import argparse


def main(args):
    data_list = [np.load(in_file) for in_file in args.input_files]
    merged = {key:np.concatenate([data_set[key] for data_set in data_list]) for key in data_list[0].keys()}
    np.savez(args.output_path, **merged)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='concatenate a list of .npz files by axis 0 for all elements')
    parser.add_argument('-i', '--input_files', nargs='+', 
                        help='list of input file path to concatenate') 
    parser.add_argument('-o', '--output_path', type=str, default='output.npz', 
                        help='output file name') 
    args = parser.parse_args()
    
    main(args)



