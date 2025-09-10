# ManuAI Inference - Refactored for single audio file processing
# Provide user with confidence scores for each prediction
import subprocess
import torch
import numpy as np
import librosa
from PIL import Image
import torchaudio.transforms as T
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from PIL import Image, ImageOps
import tempfile
from peft import PeftModel
import tkinter as tk
from tkinter import filedialog
#from tqdm import tqdm
import os

def assess_audio_quality(
    audio,
    sr,
    min_snr=14.0,
    max_silence_ratio=0.5,
    min_spectral_centroid=500,
    max_spectral_centroid=8000,
    max_zcr=0.3,
    quality_pass_score=65,
):
    """
    Advanced audio quality assessment.
    Uses weighted scoring across multiple features:
      - Signal-to-Noise Ratio (SNR)
      - Silence ratio
      - Spectral centroid
      - Zero Crossing Rate (ZCR)

    Returns True if quality score passes threshold, else False.
    """
    try:
        quality_score = 0

        # --- SNR estimation ---
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        if len(rms) < 10:
            return False

        rms_sorted = np.sort(rms)
        noise_level = np.mean(rms_sorted[: len(rms_sorted) // 10])
        signal_level = np.mean(rms_sorted[-len(rms_sorted) // 10:])
        snr_db = 20 * np.log10(signal_level / (noise_level + 1e-10))
        if snr_db >= min_snr:
            quality_score += 30

        # --- Silence detection ---
        threshold = np.mean(rms) * 0.1
        silence_frames = np.sum(rms < threshold)
        silence_ratio = silence_frames / len(rms)
        if silence_ratio <= max_silence_ratio:
            quality_score += 25

        # --- Spectral centroid ---
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
        if min_spectral_centroid <= centroid <= max_spectral_centroid:
            quality_score += 25

        # --- Zero Crossing Rate ---
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0])
        if zcr <= max_zcr:
            quality_score += 20

        return quality_score >= quality_pass_score
    except Exception:
        return False

def convert_to_mel_spectrogram(sample,
                               image_size=224,
                               target_frames=256,
                               segment_len=4.0,
                               spectrogram_mode="log-mel"):
    """
    Convert audio sample to Mel spectrogram for ViT-based bird sound classification.

    """
    audio = sample["audio"]
    waveform = torch.tensor(audio["array"], dtype=torch.float32)
    sample_rate = audio["sampling_rate"]

    # Ensure waveform is 2D (1, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    hop_length = (segment_len * sample_rate) // target_frames
    hop_length = 2 ** int(np.floor(np.log2(hop_length)))
    n_fft = hop_length * 4
    n_mels = 224

    # Configure Mel spectrogram for bird sounds
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        f_min=100,
        f_max=10000,
    )

    # Generate Mel spectrogram
    mel_spec = mel_spec_transform(waveform).squeeze(0).numpy()
    y = librosa.power_to_db(mel_spec, ref=np.max)
    sample["log_mel"] = y

    # Process spectrogram based on mode
    if spectrogram_mode == "log-mel":
        # Convert to log scale (decibels)
        y = np.clip(y, -80, 0)
        y = ((y + 80) / 80 * 255).astype(np.uint8)
        y = np.stack([y] * 3, axis=-1)  # (n_mels, time, 3)
    elif spectrogram_mode == "delta3":
        # Log-Mel with delta and delta-delta features
        delta = librosa.feature.delta(y)
        delta2 = librosa.feature.delta(y, order=2)
        y = np.stack([y, delta, delta2], axis=-1)  # (n_mels, time, 3)
    else:
        raise ValueError(f"Unknown spectrogram mode: {spectrogram_mode}")

    # Convert to RGB image
    img = Image.fromarray(y).convert("RGB")
    img = ImageOps.pad(img, (image_size, image_size), color=(0, 0, 0))  # pad to 224x224 without stretching
    
    sample["image"] = img
    return sample

def convert_to_wav(input_path, output_path=None, sr=22050):
    """
    Converts any audio/video file to WAV format with specified sample rate using ffmpeg.
    Returns path to the converted WAV file.
    """
    if output_path is None:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        output_path = temp_wav.name

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i", input_path,
        "-ar", str(22050),  # set sample rate
        "-ac", "1",      # mono
        "-vn",           # no video
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except Exception as e:
        print(f"Error converting {input_path} to WAV with ffmpeg: {e}")
        return None
    
def process_audio(
    audio_file,
    duration=4.0,
    quality_filter=False,
    noise_reduction=False,
    sr=22050,
):
    """
    Segment a single audio file into fixed-duration chunks, apply quality filtering,
    and convert each segment into a mel spectrogram for inference.
    """
    segments = []
    try:
        audio, sr = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print(f"❌ Error loading {audio_file}: {e}")
        return []

    chunk_size = int(duration * sr)
    hop_size = chunk_size // 2

    # --- optional noise reduction ---
    if noise_reduction:
        audio = apply_noise_reduction(audio, sr)

    # --- sliding window segmentation ---
    for start in range(0, len(audio) - chunk_size + 1, hop_size):
        chunk = audio[start : start + chunk_size]

        # skip very quiet chunks
        if np.max(np.abs(chunk)) < 0.005:
            continue

        # pad/trim to exact chunk size
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        elif len(chunk) > chunk_size:
            chunk = chunk[:chunk_size]

        # optional quality filter
        if quality_filter and not assess_audio_quality(chunk, sr):
            continue

        segments.append(chunk)

    # --- fallback: take center chunk if nothing passed ---
    if not segments and len(audio) >= chunk_size:
        start = (len(audio) - chunk_size) // 2
        chunk = audio[start : start + chunk_size]

        if np.max(np.abs(chunk)) >= 0.005:
            if (not quality_filter) or assess_audio_quality(chunk, sr):
                segments.append(chunk)

    # --- convert to spectrograms ---
    mel_spectrograms = []
    for seg in segments:
        sample = {"audio": {"array": seg, "sampling_rate": sr}}
        mel_sample = convert_to_mel_spectrogram(sample)
        mel_spectrograms.append(mel_sample["image"])

    print(f"✅ Processed {len(mel_spectrograms)} segments from {audio_file}")
    return mel_spectrograms


def apply_noise_reduction(y, sr, factor=0.8):
    """
    Simple spectral gating noise reduction.
    Matches the training-side implementation.
    """
    y = y - np.mean(y)
    stft = librosa.stft(y, n_fft=1024, hop_length=256)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_profile = np.percentile(mag, 10, axis=1, keepdims=True)
    mask = mag >= noise_profile
    mag_denoised = mag * (mask + (1 - mask) * factor)
    stft_denoised = mag_denoised * np.exp(1j * phase)
    return librosa.istft(stft_denoised, hop_length=256).astype(np.float32)

def save_mel_spectrograms(mel_spectrograms, output_dir="./inference_outputs"):
    """
    Save mel spectrogram images to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, mel_spec in enumerate(mel_spectrograms):
        output_path = os.path.join(output_dir, f"mel_spec_{i}.png")
        mel_spec.save(output_path)
        #print(f"Saved mel spectrogram to {output_path}")

def load_model(model_path, lora_adapter_path=None):
    """
    Load the fine-tuned model and processor. Returns (model, processor, labels_from_model)
    """
    try:
        # Load config to get num_labels, id2label, label2id
        config = ViTConfig.from_pretrained(model_path)
        print(f"Loaded config: num_labels={config.num_labels}, id2label={config.id2label}")
        num_labels = config.num_labels
        id2label = config.id2label
        label2id = config.label2id

        model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        processor = ViTImageProcessor.from_pretrained(model_path)
        if lora_adapter_path:
            model = PeftModel.from_pretrained(model, lora_adapter_path)

        labels_from_model = [id2label[str(i)] if str(i) in id2label else f"CLASS_{i}" for i in range(num_labels)]
        print(f"Model loaded from {model_path} with classes: {labels_from_model}")
        return model, processor, labels_from_model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None, None
    
def predict(model, processor, mel_spectrograms, batch_size=32, device=None, return_all_probs=False):
    """
    Perform inference on the mel spectrograms using the loaded model.
    Returns predictions as a list of (predicted_class, confidence_score)
    """
    if model is None or processor is None:
        print("Model or processor not loaded. Cannot perform inference.")
        return [] if not return_all_probs else ([], None)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    all_prob_chunks = []

    with torch.inference_mode():
        for i in range(0, len(mel_spectrograms), batch_size):
            batch_imgs = mel_spectrograms[i:i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)  # [B, num_classes]
            top_conf, top_idx = probs.max(dim=-1)
            predictions.extend(list(zip(top_idx.tolist(), top_conf.tolist())))
            if return_all_probs:
                all_prob_chunks.append(probs.cpu())

    if return_all_probs:
        if all_prob_chunks:
            all_probs = torch.cat(all_prob_chunks, dim=0)
        else:
            # Create an empty tensor with the correct number of classes from the model config
            num_classes = getattr(getattr(model, "config", object()), "num_labels", 0)
            all_probs = torch.empty((0, num_classes)) if num_classes > 0 else None
        return predictions, all_probs
    return predictions

def main(audio_file, model_path, lora_adapter_path, output_dir):
    """
    Main function to process a single audio file and perform inference.
    """
    mel_spectrograms = process_audio(audio_file, duration=3.0, quality_filter=True)
    if not mel_spectrograms:
        print("No valid segments found for inference.")
        return

    save_mel_spectrograms(mel_spectrograms, output_dir)

    model, processor, labels = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    predictions, all_probs = predict(model, processor, mel_spectrograms, batch_size=32, return_all_probs=True)

    for i, (predicted_class, confidence_score) in enumerate(predictions):
        bird_name = labels[predicted_class]
        print(f"Segment {i}: Predicted Class: {bird_name}, Confidence Score: {confidence_score:.4f}")

    if all_probs is not None and all_probs.numel() > 0:
        mean_probs = all_probs.mean(dim=0)
        overall_confidence_score, overall_predicted_class = torch.max(mean_probs, dim=0)
        overall_bird_name = labels[int(overall_predicted_class)]
        print(f"Overall Prediction: {overall_bird_name}, Confidence Score: {overall_confidence_score.item():.4f}")

        # If you prefer a stricter combination, use summed log-probabilities instead:
        total_log_probs = torch.log(all_probs + 1e-12).sum(dim=0)
        overall_predicted_class = int(torch.argmax(total_log_probs))
        overall_bird_name = labels[overall_predicted_class]
        overall_probs = torch.softmax(total_log_probs, dim=0)
        overall_confidence_score = overall_probs[overall_predicted_class].item()
        print(f"Overall Prediction (log-sum): {overall_bird_name}, Confidence Score: {overall_confidence_score:.4f}")

if __name__ == "__main__":
    model_path = "./manuai_checkpoints"
    recordings_dir = "unknown_recordings"
    lora_adapter_path = "./manuai_lora_adapter"  # Optional: for LoRA adapter
    output_dir = "./inference_outputs"

    # Open file dialog to select audio file
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    audio_file = filedialog.askopenfilename(
        title="Select an audio/video file for inference",
        filetypes=[
            ("Audio/Video files", "*.wav *.mp3 *.mp4 *.mov *.flac *.ogg *.m4a *.avi *.mkv"),
            ("All files", "*.*")
        ]
    )
    if not audio_file:
        print("No file selected. Exiting.")
        exit(1)

    print(f"Selected file: {audio_file}")

    # Convert to WAV 22050Hz if needed
    wav_file = convert_to_wav(audio_file)
    if not wav_file:
        print("Failed to convert file for processing. Exiting.")
        exit(1)

    # Ensure output directory exists and is clean
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        for f in os.listdir(output_dir):
            fp = os.path.join(output_dir, f)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except Exception:
                pass

        main(wav_file, model_path, lora_adapter_path, output_dir)
