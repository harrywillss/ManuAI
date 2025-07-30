'''
Main pipeline for ManuAI: NZ Bird Sound Classification
'''

import os
import sys
import download_data as download
import preprocess_data as preprocess

def main():
    # Step 1: Download the dataset
    print("Downloading dataset...")
    download.main()

    # Step 2: Preprocess the data
    print("Preprocessing data...")
    preprocess.main()

    # Step 3: Train the model
    print("Training model...")
    #train.train_model()

    # Step 4: Evaluate the model
    print("Evaluating model...")
    #evaluate.evaluate_model()