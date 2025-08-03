import download_data
import preprocess_data
import os

# To run notebook cells from here, you can use:
# !python -m ipykernel_launcher -f /path/to/your/notebook.ipynb
# This will execute the main function in the script
# and download the data, then preprocess it.


def main():
    # Download the audio data
    download_data.main()
    preprocess_data.main()


if __name__ == "__main__":
    main()
    print("Data download and preprocessing completed successfully.")