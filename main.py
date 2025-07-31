import download_data
import preprocess_data

def main():
    # Download the audio data
    download_data.main()
    preprocess_data.main()

if __name__ == "__main__":
    main()
    print("Data download and preprocessing completed successfully.")