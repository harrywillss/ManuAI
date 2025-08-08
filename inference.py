# ManuAI Inference - Refactored for single audio file processing
# Provide user with confidence scores for each prediction
import torch
import numpy as np
import librosa
from PIL import Image
import torchaudio.transforms as T
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import soundfile as sf
#from tqdm import tqdm
import os

labels = [
    "Saddleback/Tīeke", "Tomtit/Miromiro", "Whitehead/Pōpokotea", 
    "Morepork/Ruru", "Fantail/Pīwakawaka", "Kākā", "Tūi", 
    "Robin/Toutouwai", "Silvereye/Tauhou", "Bellbird/Korimako",
]

def assess_audio_quality(audio, sr, min_snr=10.0, max_silence_ratio=0.7, min_spectral_centroid=500, max_spectral_centroid=8000):
    """
    Assess the quality of an audio segment based on SNR, silence, and spectral centroid.
    Returns True if quality is sufficient, else False.
    """
    try:
        # SNR estimation
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        if len(rms) < 10:
            return False
        rms_sorted = np.sort(rms)
        noise_level = np.mean(rms_sorted[:len(rms_sorted)//10])
        signal_level = np.mean(rms_sorted[-len(rms_sorted)//10:])
        snr_db = 20 * np.log10(signal_level / max(noise_level, 1e-10))
        # Silence detection
        threshold = np.mean(rms) * 0.1
        silence_frames = np.sum(rms < threshold)
        silence_ratio = silence_frames / len(rms)
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        avg_spectral_centroid = np.mean(spectral_centroids)
        # Quality checks
        if snr_db < min_snr:
            return False
        if silence_ratio > max_silence_ratio:
            return False
        if not (min_spectral_centroid <= avg_spectral_centroid <= max_spectral_centroid):
            return False
        return True
    except Exception:
        return False


def convert_to_mel_spectrogram(sample):
    """
    Convert audio sample to mel spectrogram and normalize it for visualization.
    """
    if "audio" not in sample or "array" not in sample["audio"]:
        raise ValueError("Sample must contain 'audio' with 'array' key.")
    waveform = torch.tensor(sample["audio"]["array"]).unsqueeze(0)
    sample_rate = sample["audio"]["sampling_rate"]
    if sample_rate != 44100:
        resampler = T.Resample(sample_rate, 44100)
        waveform = resampler(waveform)
        sample_rate = 44100
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        n_mels=128,
        f_min=50,
        f_max=sample_rate // 2
    )
    mel_spec = mel_spectrogram_transform(waveform)
    amplitude_to_db = T.AmplitudeToDB()
    mel_spec_db = amplitude_to_db(mel_spec)
    mel_spec_np = mel_spec_db.squeeze().numpy()
    normalized = ((mel_spec_np - mel_spec_np.min()) /
                  (mel_spec_np.max() - mel_spec_np.min()) * 255).astype(np.uint8)
    rgb_image = np.stack([normalized] * 3, axis=-1)
    sample["image"] = Image.fromarray(rgb_image)
    return sample

def process_single_audio_for_inference(audio_file, duration=3.0, quality_filter=False, sr=44100):
    """
    Segment a single audio file and convert segments to mel spectrograms for inference.
    """
    segments = []
    try:
        audio, sr = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return []
    chunk_size = int(duration * sr)
    hop_size = chunk_size // 2
    for start in range(0, len(audio) - chunk_size + 1, hop_size):
        chunk = audio[start:start + chunk_size]
        if len(chunk) == chunk_size and np.max(np.abs(chunk)) > 0.01:
            if quality_filter:
                if assess_audio_quality(chunk, sr):
                    segments.append(chunk)
            else:
                segments.append(chunk)
    if not segments and len(audio) >= chunk_size:
        start = (len(audio) - chunk_size) // 2
        chunk = audio[start:start + chunk_size]
        if np.max(np.abs(chunk)) > 0.01:
            if quality_filter:
                if assess_audio_quality(chunk, sr):
                    segments.append(chunk)
            else:
                segments.append(chunk)
    mel_spectrograms = []
    for seg in segments:
        sample = {"audio": {"array": seg, "sampling_rate": sr}}
        mel_sample = convert_to_mel_spectrogram(sample)
        mel_spectrograms.append(mel_sample["image"])
    print(f"Processed {len(mel_spectrograms)} segments from {audio_file}")
    return mel_spectrograms

def save_mel_spectrograms(mel_spectrograms, output_dir):
    """
    Save mel spectrogram images to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, mel_spec in enumerate(mel_spectrograms):
        output_path = os.path.join(output_dir, f"mel_spec_{i}.png")
        mel_spec.save(output_path)
        #print(f"Saved mel spectrogram to {output_path}")

def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    """
    try:
        model = ViTForImageClassification.from_pretrained(model_path, num_labels=10)
        processor = ViTImageProcessor.from_pretrained(model_path)
        print(f"Model loaded from {model_path}")
        return model, processor
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None
    
def predict(model, processor, mel_spectrograms):
    """
    Perform inference on the mel spectrograms using the loaded model.
    """
    if model is None or processor is None:
        print("Model or processor not loaded. Cannot perform inference.")
        return []
    
    predictions = []
    for mel_spec in mel_spectrograms:
        inputs = processor(images=mel_spec, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence_scores, predicted_class = torch.max(probs, dim=-1)
        predictions.append((predicted_class.item(), confidence_scores.item()))
    
    return predictions

def main(audio_file, model_path, output_dir):
    """
    Main function to process a single audio file and perform inference.
    """
    mel_spectrograms = process_single_audio_for_inference(audio_file, duration=3.0, quality_filter=True)
    if not mel_spectrograms:
        print("No valid segments found for inference.")
        return
    
    save_mel_spectrograms(mel_spectrograms, output_dir)
    
    model, processor = load_model(model_path)
    predictions = predict(model, processor, mel_spectrograms)
    
    for i, (predicted_class, confidence_score) in enumerate(predictions):
        bird_name = labels[predicted_class]
        print(f"Segment {i}: Predicted Class: {bird_name}, Confidence Score: {confidence_score:.4f}")
    # Print most likely overall prediction
    if predictions:
        overall_predicted_class = max(predictions, key=lambda x: x[1])[0]
        overall_bird_name = labels[overall_predicted_class]
        overall_confidence_score = max(predictions, key=lambda x: x[1])[1]
        print(f"Overall Prediction: {overall_bird_name}, Confidence Score: {overall_confidence_score:.4f}")

if __name__ == "__main__":
    model_path = "vit-base-manuai"
    recordings_dir = "unknown_recordings"

    audio_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
    if not audio_files:
        print("No audio files found in the specified directory.")
        exit(1)
    
    audio_file = os.path.join(recordings_dir, audio_files[11])
    # If you want to process a specific file, you can set it directly:
    # audio_file = "path/to/your/audio.wav"
    output_dir = "output"
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(audio_file, model_path, output_dir)
# This code processes a single audio file, converts it to mel spectrograms, and performs inference using a pre-trained model.
# It provides confidence scores for each prediction and saves the mel spectrogram images to the specified output directory.
# Ensure you have the necessary libraries installed: transformers, torchaudio, librosa, PIL, numpy, tqdm.