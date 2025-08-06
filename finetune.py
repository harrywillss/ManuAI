# Script to LoRA fine-tune a ViT model using Hugging Face Transformers
# This script is designed for the ManuAI project, using the New Zealand Bird dataset.
from transformers import ViTImageProcessor
from datasets import load_dataset, Audio
import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Finetune: 
    def __init__(self):
        self.model_name = "google/vit-base-patch16-224-in21k"  # Pre-trained model to fine-tune
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.batch_size = 64  # Set batch size for training
        self.training_args = {
            "output_dir": "./model-checkpoints",  # Directory to save model checkpoints
            "remove_unused_columns": False,
            "learning_rate": 2e-4,  # Learning rate for fine-tuning
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": 16,  # Gradient accumulation steps to simulate larger batch size
            "fp16": False,  # Disable mixed precision training
            "bf16": False,  # Disable bfloat16 training
            "no_cuda": True,  # Disable CUDA to avoid GPU memory issues
            "logging_steps": 10,
            "load_best_model_at_end": True,
            "metric_for_best_model": "accuracy",
            "greater_is_better": True, 
            "evaluation_strategy": "epoch",  # Evaluate at the end of each epoch
            "save_strategy": "epoch",  # Save model at the end of each epoch
            "save_total_limit": 3,  # Limit the number of saved checkpoints
            "label_names": ["labels"],
            "warmup_ratio": 0.1,  # Warmup ratio for learning rate scheduler
            "weight_decay": 0.01,  # Weight decay for regularization
            "dataloader_num_workers": 0,  # Number of workers for data loading
            "report_to": None,  # Disable TensorBoard
            "dataloader_pin_memory": False,  # Disable pinning memory for DataLoader
            "disable_tqdm": False,  # Explicitly set to False to ensure it's not suppressed
        }

    def view_sample(self, sample):
        """
        View a sample from the dataset.
        """
        if "image" in sample:
            image = sample["image"]
            if isinstance(image, Image.Image):
                image.show()  # Display the image
            else:
                logging.warning("Sample does not contain a valid image.")

    def convert_to_mel_spectrogram(self, sample):
        """
        Convert audio data into Mel Spectrograms using torchaudio.
        """
        # Load audio using torchaudio instead of librosa
        waveform, sample_rate = torchaudio.load(sample["audio"]["path"])
        
        # Resample if needed
        if sample_rate != 44100:
            resampler = T.Resample(sample_rate, 44100)
            waveform = resampler(waveform)
            sample_rate = 44100
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Create mel spectrogram transform
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            n_mels=128,
            f_min=50,
            f_max=sample_rate // 2
        )
        
        # Apply transform
        mel_spec = mel_spectrogram_transform(waveform)
        
        # Convert to dB scale
        amplitude_to_db = T.AmplitudeToDB()
        mel_spec_db = amplitude_to_db(mel_spec)
        
        # Convert to numpy and normalize
        mel_spec_np = mel_spec_db.squeeze().numpy()
        normalized = ((mel_spec_np - mel_spec_np.min()) / 
                    (mel_spec_np.max() - mel_spec_np.min()) * 255).astype(np.uint8)
        
        # Convert to RGB image
        rgb_image = np.stack([normalized] * 3, axis=-1)
        sample["image"] = Image.fromarray(rgb_image)
        return sample

    def _extract_class(self, sample):
        """
        Extract classes from the dataset.
        The classes are in the file names, so we need to parse them.
        The file names are formatted as '{bird_id}_{english_name}_{scientific_name}_{scientific_subspecie}_{song}_segment_{num}.wav'.
        """
        # Extract the class from the file name
        audio_path = sample["audio"]["path"]
        class_name = audio_path.split("/")[-1].split("_")[1]  # The class is the second part of the file name (the English name)
        sample["class"] = class_name
        return sample

    def transform(self, sample):
        """
        Transform the sample to prepare it for the model.
        """   
        inputs = self.processor(
            [x for x in sample["image"]],
            return_tensors="pt",
            do_normalize=True, # Normalize the images
        )
        inputs["label"] = sample["class"]
        return inputs
    
    def load_data(self, dataset_path="segments/"):
        """
        Load the dataset from the specified path and convert audio data into Mel Spectrograms.
        Directory structure:
        segments/
        ├── {english_name}
        │   ├── {scientific_name}_{scientific_subspecie}
        │   │   ├── {english_name}_{scientific_name}_{scientific_subspecie}_{song}_segment_{num}.wav
        """

        # Load the dataset from path as "audio" files
        dataset = load_dataset(
            "audiofolder",
            data_dir=dataset_path,
            split="train",
            #cache_dir="./cache",  # Cache directory to speed up loading
            #use_auth_token=True,  # Use authentication token if required
        )

        # Cast audio column to use soundfile backend
        dataset = dataset.cast_column("audio", Audio(decode=False))
        logging.info(f"Dataset loaded with {len(dataset)} samples from {dataset_path}")

        if not dataset:
            raise ValueError("Dataset could not be loaded. Check the dataset path and structure.")
        if "audio" not in dataset.column_names:
            raise ValueError("Dataset does not contain 'audio' column. Check the dataset structure.")

        logger.info(f"Dataset loaded with {len(dataset)} samples.")
        logger.info(f"Sample audio file: {dataset[0]['audio']['path']}")

        # Extract labels from the dataset
        dataset = dataset.map(
            self.extract_label,
            remove_columns=["label"],
            desc="Extracting labels from audio file names",
            num_proc=2,  # Use 2 processes for faster label extraction
        )

        if not dataset[0]["class"]:
            raise ValueError("No labels extracted from the dataset. Check the label extraction logic.")
        logging.info(f"Labels extracted: {dataset[0]['class']}")

        # Convert audio data into Mel Spectrograms
        dataset = dataset.map(
            self.convert_to_mel_spectrogram, 
            remove_columns=["audio"],
            num_proc=4,  # Use 4 processes for faster conversion
            desc="Converting audio to Mel Spectrograms"
        )
        # Check if Mel Spectrograms are created correctly
        if "image" not in dataset.column_names:
            raise ValueError("Mel Spectrograms were not created correctly. Check the conversion logic.")
        logging.info(f"Converted Mel Spectrograms for {len(dataset)} samples.")

        # Transform the dataset for the model
        dataset = dataset.with_transform(self.transform, num_proc=4)  # Use 4 processes for faster transformation
        if not dataset[0]["image"]:
            raise ValueError("Dataset transformation failed. Check the transformation logic.")
        
        # Ensure the dataset is ready for model input
        dataset.set_format(type="torch", columns=["image", "label"])
        logging.info("Dataset transformed for model input.")

        return dataset


def main():
    # TESTING
    finetune = Finetune()
    dataset = finetune.load_data()


if __name__ == "__main__":
    main()