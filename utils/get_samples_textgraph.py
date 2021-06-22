#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:11:34 2021

@author: poojaoza
"""

import argparse
import csv


def get_samples(input_file):
    positive_samples = 0
    negative_samples = 0
    with open(input_file, 'r') as f:
        rd = csv.reader(f, delimiter="\t", quotechar='"')
        for row in rd:
            #print(row)
            if row[3] == "0":
                negative_samples += 1
            else:
                positive_samples += 1
    print(negative_samples)
    print(positive_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the positive and negative sample numbers")
    parser.add_argument('--inputfile')
    args = parser.parse_args()
    get_samples(args.inputfile)