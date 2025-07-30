'''
Main pipeline for ManuAI: NZ Bird Sound Classification
This script orchestrates the entire process from downloading the dataset to preprocessing it.
It is designed to be run from the root directory of the project.
Make sure to have the necessary modules available in the same directory or installed in your Python environment.
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

if __name__ == "__main__":
    # Ensure the script is run from the root directory
    if not os.path.exists("download_data.py") or not os.path.exists("preprocess_data.py"):
        print("Error: This script must be run from the root directory of the project.")
        sys.exit(1)    
    main()
    print("Pipeline completed successfully.")
    sys.exit(0)
