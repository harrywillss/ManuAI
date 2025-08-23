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
    
def pcen_custom(spec, s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    """
    Simple PCEN implementation for log-mel spectrograms.
    spec: (1, n_mels, time) torch.Tensor
    """
    M = torch.zeros_like(spec)
    M[..., 0] = spec[..., 0]
    for t in range(1, spec.shape[-1]):
        M[..., t] = (1 - s) * M[..., t - 1] + s * spec[..., t]
    pcen = (spec / (eps + M) ** alpha + delta) ** r - delta ** r
    return pcen


def convert_to_mel_spectrogram(sample, mode="rgb_copy"):
    """
    Convert audio to spectrogram image for ViT input.

    mode options:
        "rgb_copy"   -> log-mel copied into RGB channels
        "stack_3ch"  -> 3-channel stack: [log-mel, PCEN, delta]
    """
    waveform = torch.tensor(sample["audio"]["array"]).unsqueeze(0)
    sample_rate = sample["audio"]["sampling_rate"]

    # Resample
    if sample_rate != 22050:
        waveform = T.Resample(sample_rate, 22050)(waveform)
        sample_rate = 22050

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=296,
        n_mels=128,
        f_min=150,
        f_max=10000
    )
    mel_spec = mel_transform(waveform)

    # Transformations
    logmel = T.AmplitudeToDB()(mel_spec)
    pcen = pcen_custom(logmel)  # Custom PCEN implementation
    delta = T.ComputeDeltas()(logmel)

    def pad_to_square(spec):
        pad_freq = 224 - spec.shape[1]
        pad_top = pad_freq // 2
        pad_bottom = pad_freq - pad_top
        # Pad frequency axis: (left, right, top, bottom)
        return F.pad(spec, (0, 0, pad_top, pad_bottom), value=0)

    logmel = pad_to_square(logmel)
    pcen = pad_to_square(pcen)
    delta = pad_to_square(delta)

    if mode == "rgb_copy":
        # normalize logmel to 0-255
        lm_np = logmel.squeeze().numpy()
        lm_norm = ((lm_np - lm_np.min()) / (lm_np.max() - lm_np.min()) * 255).astype(np.uint8)
        rgb_image = np.stack([lm_norm] * 3, axis=-1)

    elif mode == "stack_3ch":
        def norm_channel(ch):
            ch_np = ch.squeeze().numpy()
            return ((ch_np - ch_np.min()) / (ch_np.max() - ch_np.min()) * 255).astype(np.uint8)

        logmel_np = norm_channel(logmel)
        pcen_np = norm_channel(pcen)
        delta_np = norm_channel(delta)

        rgb_image = np.stack([logmel_np, pcen_np, delta_np], axis=-1)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    sample["image"] = Image.fromarray(rgb_image)
    return sample

def process_audio(audio_file, duration=3.0, quality_filter=False, sr=44100):
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
    Load the fine-tuned model and processor. Returns (model, processor, labels_from_model)
    """
    try:
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(model_path)

        # Build ordered labels robustly from config
        labels_from_model = None

        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict) and id2label:
            try:
                # Normalize keys to int (handles both int and "0"-style string keys)
                norm = {int(k): v for k, v in id2label.items()}
                labels_from_model = [norm[i] for i in range(model.config.num_labels)]
            except Exception:
                labels_from_model = None  # Fallback to other sources

        if labels_from_model is None:
            label2id = getattr(model.config, "label2id", None)
            if isinstance(label2id, dict) and label2id:
                # Create list where index is class id
                labels_from_model = [None] * model.config.num_labels
                for label, idx in label2id.items():
                    try:
                        idx = int(idx)
                        if 0 <= idx < model.config.num_labels:
                            labels_from_model[idx] = str(label)
                    except Exception:
                        pass
                # Fill any gaps
                for i in range(model.config.num_labels):
                    if not labels_from_model[i]:
                        labels_from_model[i] = f"CLASS_{i}"

        if labels_from_model is None:
            labels_from_model = [f"CLASS_{i}" for i in range(model.config.num_labels)]

        print(f"Model loaded from {model_path} with classes: {labels_from_model}")
        return model, processor, labels_from_model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None, None
    
def predict(model, processor, mel_spectrograms, batch_size=16, device=None, return_all_probs=False):
    """
    Perform inference on the mel spectrograms using the loaded model.
    Returns:
      - predictions: list of (pred_class_idx, top_conf) per segment
      - all_probs (optional): tensor of shape [num_segments, num_classes] with per-segment probabilities
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

def main(audio_file, model_path, output_dir):
    """
    Main function to process a single audio file and perform inference.
    """
    mel_spectrograms = process_audio(audio_file, duration=3.0, quality_filter=True)
    if not mel_spectrograms:
        print("No valid segments found for inference.")
        return

    save_mel_spectrograms(mel_spectrograms, output_dir)

    model, processor, labels_from_model = load_model(model_path)
    if model is None:
        return

    predictions, all_probs = predict(model, processor, mel_spectrograms, batch_size=16, return_all_probs=True)

    for i, (predicted_class, confidence_score) in enumerate(predictions):
        bird_name = labels_from_model[predicted_class]
        print(f"Segment {i}: Predicted Class: {bird_name}, Confidence Score: {confidence_score:.4f}")

    if all_probs is not None and all_probs.numel() > 0:
        mean_probs = all_probs.mean(dim=0)
        overall_confidence_score, overall_predicted_class = torch.max(mean_probs, dim=0)
        overall_bird_name = labels_from_model[int(overall_predicted_class)]
        print(f"Overall Prediction: {overall_bird_name}, Confidence Score: {overall_confidence_score.item():.4f}")

        # If you prefer a stricter combination, use summed log-probabilities instead:
        # total_log_probs = torch.log(all_probs + 1e-12).sum(dim=0)
        # overall_predicted_class = int(torch.argmax(total_log_probs))
        # overall_bird_name = labels[overall_predicted_class]
        # overall_probs = torch.softmax(total_log_probs, dim=0)
        # overall_confidence_score = overall_probs[overall_predicted_class].item()
        # print(f"Overall Prediction (log-sum): {overall_bird_name}, Confidence Score: {overall_confidence_score:.4f}")

if __name__ == "__main__":
    model_path = "vit-base-manuai"
    recordings_dir = "unknown_recordings"

    audio_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
    if not audio_files:
        print("No audio files found in the specified directory.")
        exit(1)

    while True:
        num = input(f"Enter the index of the audio file to process (0-{len(audio_files)-1}): ")
        audio_file = os.path.join(recordings_dir, audio_files[int(num)])
        # If you want to process a specific file, you can set it directly:
        # audio_file = "path/to/your/audio.wav"
        output_dir = "output"
        # Ensure output directory exists before cleaning it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Safely remove only files, ignore subdirectories
            for f in os.listdir(output_dir):
                fp = os.path.join(output_dir, f)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass

        main(audio_file, model_path, output_dir)
        