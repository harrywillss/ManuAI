'''
Main pipeline for ManuAI: NZ Bird Sound Classification
'''

import os
import sys
import download_data as download
import preprocess_data as preprocess
import train_model as train
import evaluate_model as evaluate

def main():
    # Step 1: Download the dataset
    print("Downloading dataset...")
    download.download_dataset()

    # Step 2: Preprocess the data
    print("Preprocessing data...")
    preprocess.preprocess_data()

    # Step 3: Train the model
    print("Training model...")
    train.train_model()

    # Step 4: Evaluate the model
    print("Evaluating model...")
    evaluate.evaluate_model()

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
    sys.exit(0)