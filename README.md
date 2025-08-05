# ManuAI 🦜 - New Zealand Bird Sound Classification

A machine learning project that classifies New Zealand bird sounds using Vision Transformer (ViT) models fine-tuned with LoRA (Low-Rank Adaptation) on mel spectrogram representations of audio recordings.

## 🎯 Project Overview

ManuAI transforms bird audio recordings into mel spectrograms and uses computer vision techniques to classify different New Zealand bird species. The project leverages the unique approach of treating audio classification as an image classification problem, using Google's Vision Transformer (ViT) model fine-tuned with LoRA for efficient training.

### Key Features

- **Audio-to-Image Conversion**: Converts bird audio recordings to mel spectrograms for visual processing
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning of pre-trained ViT models
- **Class Imbalance Handling**: Implements weighted loss functions to handle imbalanced datasets
- **Automated Data Pipeline**: Complete pipeline from data download to model training
- **Early Stopping**: Prevents overfitting with configurable early stopping callbacks

## 🏗️ Project Structure

```
ManuAI/
├── README.md                    # Project documentation
├── main.py                      # Main execution script
├── download_data.py             # Xeno-canto API data downloader
├── preprocess_data.py           # Audio preprocessing and segmentation
├── dont_ai.ipynb              # Main training notebook
├── finetune.ipynb              # Alternative fine-tuning notebook
├── finetune_clean.ipynb        # Clean version of fine-tuning
├── spectrogram_review.ipynb    # Spectrogram analysis and visualization
├── safetensors_to_art.py       # Model conversion utilities
├── brainstorm.md               # Project planning and ideas
├── downloads/                  # Raw audio files from Xeno-canto
├── segments/                   # Processed audio segments
├── training_data/              # Prepared training data
├── model-checkpoints/          # Model training checkpoints
├── logs/                       # Training logs
└── reports/                    # Analysis reports
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- ~10GB storage space for datasets

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/harrywillss/ManuAI.git
   cd ManuAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   Create a `.env` file for API configurations:
   ```bash
   # Add any API keys or configuration here
   ```

### Quick Start

1. **Download and preprocess data**
   ```bash
   python main.py
   ```

2. **Train the model**
   Open `dont_ai.ipynb` in Jupyter and run all cells, or use the command line:
   ```bash
   jupyter notebook dont_ai.ipynb
   ```

## 📊 Dataset

The project uses bird recordings from [Xeno-canto](https://xeno-canto.org/), a citizen science project focused on sharing bird sounds from around the world.

### Data Processing Pipeline

1. **Download**: Fetches New Zealand bird recordings via Xeno-canto API
2. **Segmentation**: Splits recordings into 4-second segments
3. **Quality Filtering**: Removes low-quality or silent segments
4. **Spectrogram Conversion**: Converts audio to mel spectrograms
5. **Augmentation**: Applies data augmentation techniques

### Supported Bird Species

The model currently supports classification of major New Zealand bird species including:
- Tui (*Prosthemadera novaeseelandiae*)
- Bellbird (*Anthornis melanura*)
- Kaka (*Nestor meridionalis*)
- Robin (*Petroica* species)
- Morepork (*Ninox novaeseelandiae*)
- Fantail (*Rhipidura fuliginosa*)
- And many more...

## 🤖 Model Architecture

### Base Model
- **Google ViT-Base-Patch16-224**: Pre-trained Vision Transformer
- **Input Size**: 224x224 RGB images (mel spectrograms)
- **Patch Size**: 16x16 pixels

### LoRA Configuration
```python
LoraConfig(
    r=16,                           # Rank of adaptation
    lora_alpha=16,                  # Scaling factor
    target_modules=["query", "value"], # Target attention modules
    lora_dropout=0.1,               # Dropout rate
    bias="none",                    # No bias adaptation
    modules_to_save=["classifier"]  # Save classifier head
)
```

### Training Features
- **Class Weighting**: Handles imbalanced datasets
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Warmup and decay
- **Mixed Precision**: Optional FP16 training

## 📈 Performance

The model achieves competitive performance on New Zealand bird sound classification:

- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~80-85%
- **Model Size**: ~22MB (LoRA adapter only)
- **Inference Time**: <100ms per sample

## 🛠️ Usage

### Training a New Model

```python
# Configure training parameters
batch_size = 64
epochs = 10
learning_rate = 2e-4

# Run training
python main.py
```

### Making Predictions

```python
from transformers import AutoImageProcessor
from peft import PeftModel

# Load model and processor
model = PeftModel.from_pretrained(base_model, "path/to/lora/adapter")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Predict on new spectrogram
prediction = predict_image_class(spectrogram_image, model, processor)
print(f"Predicted bird species: {prediction}")
```

## 🔧 Configuration

### Audio Processing Parameters
```python
# Mel spectrogram settings
n_mels = 128          # Number of mel bands
n_fft = 2048          # FFT window size
hop_length = 512      # Hop length for STFT
sr = 44100            # Sample rate
fmin = 50             # Minimum frequency
fmax = 22050          # Maximum frequency (Nyquist)
```

### Training Parameters
```python
# Training configuration
batch_size = 64
learning_rate = 2e-4
warmup_ratio = 0.1
weight_decay = 0.01
gradient_accumulation_steps = 4
```

## 📋 Requirements

### Core Dependencies
```
torch>=1.9.0
transformers>=4.21.0
datasets>=2.0.0
peft>=0.3.0
librosa>=0.9.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
requests>=2.26.0
soundfile>=0.10.0
Pillow>=8.3.0
```

### Optional Dependencies
```
jupyter>=1.0.0        # For notebook execution
python-dotenv>=0.19.0 # For environment variables
tensorboard>=2.7.0    # For training visualization
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Xeno-canto](https://xeno-canto.org/) for providing the bird sound database
- [Hugging Face](https://huggingface.co/) for the Transformers library and model hosting
- [Google Research](https://github.com/google-research/vision_transformer) for the Vision Transformer architecture
- Microsoft for the LoRA (Low-Rank Adaptation) technique

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{manuai2024,
  title={ManuAI: New Zealand Bird Sound Classification using Vision Transformers},
  author={Harry Wills},
  year={2024},
  url={https://github.com/harrywillss/ManuAI}
}
```

## 📞 Contact

Harry Wills - [@harrywillss](https://github.com/harrywillss)

Project Link: [https://github.com/harrywillss/ManuAI](https://github.com/harrywillss/ManuAI)

---

*Made with ❤️ for New Zealand's native bird conservation*

machine-learning computer-vision audio-classification vision-transformer lora fine-tuning bird-sounds new-zealand conservation mel-spectrogram pytorch transformers huggingface xeno-canto ornithology