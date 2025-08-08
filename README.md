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
├── main.py                      # Main execution script for downloading data & preprocessing
├── download_data.py             # Xeno-canto API data downloader
├── preprocess_data.py           # Audio preprocessing and segmentation
├── lora-finetune.ipynb        # LoRA finetuning of model
├── brainstorm.md               # Project planning and ideas
├── downloads/                  # Raw audio files from Xeno-canto
├── segments/                   # Processed audio segments
├── training_data/              # Prepared training data in subfolders
├── logs/                       # Data logs
└── reports/                    # Analysis reports
```

## 🚀 Getting Started

### Prerequisites


## 🧑‍💻 How to Use

Follow these steps to train and use ManuAI:

1. **Download Data**
   - Run `download_data.py` to fetch New Zealand bird recordings from Xeno-canto.
   - Example:
     ```bash
     python download_data.py
     ```

2. **Preprocess Data**
   - Run `preprocess_data.py` to segment and convert audio files into mel spectrograms.
   - Example:
     ```bash
     python preprocess_data.py
     ```

3. **Fine-tune the Model**
   - Open and run all cells in `lora-finetune.ipynb` to fine-tune the Vision Transformer model using LoRA.

4. **Run Inference**
   - Use `inference.py` to classify new bird audio samples.
   - Example:
     ```bash
     python inference.py --input path/to/spectrogram.png
     ```

See each script/notebook for additional options and configuration details.

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
