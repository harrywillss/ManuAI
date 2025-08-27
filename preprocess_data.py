'''
Audio processing module for bird sound classification.
This module handles audio loading, segmentation, feature extraction, and data preparation for training.
'''
import os
import shutil
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from scipy.signal import medfilt
from sklearn.preprocessing import LabelEncoder

os.nice(10)  # makes the process "nicer" to your system (memory problems fr)

class AudioProcessor:
    """Handles audio processing, and quality assessment for bird sound classification."""
    
    def __init__(self, data_dir="training_data", duration=4, save_segments=True,
                 segments_dir="segments", dev_mode=False, dev_limit=10,
                 use_quality_filter=False, use_noise_reduction=False):
        self.data_dir = data_dir
        self.duration = duration
        self.save_segments = save_segments
        self.segments_dir = segments_dir
        self.dev_mode = dev_mode
        self.dev_limit = dev_limit
        self.use_quality_filter = use_quality_filter
        self.use_noise_reduction = use_noise_reduction
        self.config = {
            'target_sr': 22050,
            'min_duration': 1.0,
            'noise_gate_threshold': 0.02,
            'noise_reduce_factor': 0.6,
            'min_snr': 10.0,
            'max_silence_ratio': 0.7,
            'min_spectral_centroid': 500,
            'max_spectral_centroid': 8000,
            'max_zcr': 0.3,
            'quality_pass_score': 60
        }

    def apply_noise_reduction(self, audio, sr):
        """Apply noise reduction using spectral gating and median filtering."""
        if not self.use_noise_reduction:
            return audio
        try:
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Median filter for click/pop removal
            audio = medfilt(audio, kernel_size=3)
            
            # Spectral gating
            stft = librosa.stft(audio, hop_length=256, win_length=1024)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            sorted_mags = np.sort(magnitude.flatten())
            noise_floor = np.mean(sorted_mags[:int(len(sorted_mags) * 0.1)])
            noise_mask = magnitude < (self.config['noise_gate_threshold'] * np.max(magnitude))
            magnitude[noise_mask] *= self.config['noise_reduce_factor']
            audio = librosa.istft(magnitude * np.exp(1j * phase), hop_length=256, win_length=1024)
            
            # Normalise
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            return audio
        except Exception as e:
            print(f"Warning: Noise reduction failed: {e}")
            return audio

    def _compute_snr(self, audio, frame_length=2048, hop_length=512):
        """Compute SNR as a proxy using RMS energy."""
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        if len(rms) < 10:
            return 0, False, "Too short for analysis"
        rms_sorted = np.sort(rms)
        noise_level = np.mean(rms_sorted[:len(rms_sorted)//10])
        signal_level = np.mean(rms_sorted[-len(rms_sorted)//10:])
        snr_db = 20 * np.log10(signal_level / max(noise_level, 1e-10))
        return snr_db, snr_db >= self.config['min_snr'], None

    def _compute_silence_ratio(self, audio, frame_length=2048, hop_length=512):
        """Compute silence ratio based on RMS."""
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.mean(rms) * 0.1
        silence_ratio = np.sum(rms < threshold) / len(rms)
        return silence_ratio, silence_ratio <= self.config['max_silence_ratio'], None

    def assess_audio_quality(self, audio, sr):
        """Assess audio quality based on multiple metrics."""
        try:
            quality_score = 0
            reasons = []

            # SNR
            snr_db, snr_pass, snr_reason = self._compute_snr(audio)
            if snr_reason:
                return {"quality_score": 0, "pass": False, "reason": snr_reason}
            if snr_pass:
                quality_score += 30
            else:
                reasons.append(f"Low SNR: {snr_db:.1f}dB < {self.config['min_snr']}dB")

            # Silence ratio
            silence_ratio, silence_pass, _ = self._compute_silence_ratio(audio)
            if silence_pass:
                quality_score += 25
            else:
                reasons.append(f"Too much silence: {silence_ratio:.1%} > {self.config['max_silence_ratio']:.1%}")

            # Spectral centroid
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
            centroid_pass = self.config['min_spectral_centroid'] <= spectral_centroid <= self.config['max_spectral_centroid']
            if centroid_pass:
                quality_score += 25
            else:
                reasons.append(f"Spectral centroid out of range: {spectral_centroid:.0f}Hz")

            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            zcr_pass = zcr < self.config['max_zcr']
            if zcr_pass:
                quality_score += 20
            else:
                reasons.append(f"High zero crossing rate: {zcr:.3f} (possible noise)")

            return {
                "quality_score": quality_score,
                "pass": quality_score >= self.config['quality_pass_score'],
                "reasons": reasons
            }
        except Exception as e:
            return {"quality_score": 0, "pass": False, "reason": f"Analysis failed: {str(e)}"}

    def pad_or_trim_audio(self, audio, target_length):
        """Pad or trim audio to the target length."""
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant')
        return audio[:target_length]

    def load_and_validate_audio(self, file_path, min_duration=None, target_sr=None):
        """Load and validate an audio file, optionally resample."""
        min_duration = min_duration or self.config['min_duration']
        target_sr = target_sr or self.config['target_sr']
        try:
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr
            if duration < min_duration:
                return None, f"Duration {duration:.2f}s too short"
            if target_sr and sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            return audio, None
        except Exception as e:
            return None, str(e)

    def _discover_files(self, data_dir):
        """Discover WAV files and their labels, applying dev mode limits if enabled."""
        all_files = []
        species_counts = {} if self.dev_mode else None
        for root, _, files in os.walk(data_dir):
            wav_files = [f for f in files if f.endswith('.wav')]
            for file in wav_files:
                label = file[:-4].split('_')[1]  # Assuming file name format is "{id}_{english name}_{scientific name}_{scientific subspecie}_{call/song}.wav"
                if self.dev_mode:
                    if label not in species_counts:
                        species_counts[label] = 0
                    if species_counts[label] >= self.dev_limit:
                        continue
                    species_counts[label] += 1
                all_files.append((root, file, label))
        if self.dev_mode:
            print(f"🔧 DEV MODE: Limited to {self.dev_limit} files per species during segmentation")
            print(f"Will process {len(all_files)} files from {len(species_counts)} species")
        return all_files

    def _segment_audio(self, audio, sr, chunk_size, hop_size):
        """Segment audio into fixed-length chunks with quality filtering."""
        segments = []
        for start in range(0, len(audio) - chunk_size + 1, hop_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) == chunk_size and np.max(np.abs(chunk)) > 0.01:
                if self.use_quality_filter:
                    quality_result = self.assess_audio_quality(chunk, sr)
                    if quality_result["pass"]:
                        segments.append(self.pad_or_trim_audio(chunk, chunk_size))
                else:
                    segments.append(self.pad_or_trim_audio(chunk, chunk_size))
        return segments

    def _save_segment(self, clip, segments_dir, root, file, index, sr):
        """Save a single audio segment to the segments directory."""
        parts = file[:-4].split('_')
        if len(parts) < 4:
            print(f"⚠️ Invalid filename format: {file}")
            english_name = "unknown"
            scientific_name = "unknown"
        else:
            english_name = parts[1]  # e.g., 'tui'
            scientific_name = f"{parts[2]}_{parts[3]}"  # e.g., 'prosthemadera_novaeseelandiae'
        segment_folder = os.path.join(segments_dir, english_name, scientific_name)
        os.makedirs(segment_folder, exist_ok=True)
        segment_filename = f"{file[:-4]}_segment_{index}.wav"
        sf.write(os.path.join(segment_folder, segment_filename), clip, sr)

    def load_audio_data(self, data_dir=None, save_segments=None, segments_dir=None):
        """Load and process audio data, returning clips and labels."""
        data_dir = data_dir or self.data_dir
        save_segments = save_segments if save_segments is not None else self.save_segments
        segments_dir = segments_dir or self.segments_dir

        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist.")
            self.resample_audio()
            # Re-discover after resampling
        if os.path.exists(segments_dir) and input("Segments directory exists. Do you want to skip segmentation and load existing segments? (y/n): ").strip().lower() == 'y':
            print("Loading existing segments...")
            return self.load_existing_segments(segments_dir)

        audio_clips, labels = [], []
        all_files = self._discover_files(data_dir)
        if not all_files:
            print(f"No audio files found in {data_dir}. Please check the directory.")
            return audio_clips, labels

        if save_segments:
            os.makedirs(segments_dir, exist_ok=True)

        stats = {"processed": 0, "failed": 0, "segments": 0}
        sr = self.config['target_sr']
        chunk_size = int(self.duration * sr)
        hop_size = chunk_size // 2

        with tqdm(total=len(all_files), desc="Processing audio files", unit="file") as pbar:
            for root, file, label in all_files:
                file_path = os.path.join(root, file)
                pbar.set_postfix_str(f"Current: {file[:25]}... | Species: {label}")
                audio, error = self.load_and_validate_audio(file_path, min_duration=1.0, target_sr=sr)
                if error:
                    tqdm.write(f"⚠️ Skipping {file}: {error}")
                    stats["failed"] += 1
                else:
                    if self.use_noise_reduction:
                        audio = self.apply_noise_reduction(audio, sr)
                    segments = self._segment_audio(audio, sr, chunk_size, hop_size)
                    if segments:
                        stats["processed"] += 1
                        stats["segments"] += len(segments)
                        for i, clip in enumerate(segments):
                            if len(clip) >= chunk_size * 0.75:
                                clip = self.pad_or_trim_audio(clip, chunk_size)
                                audio_clips.append(clip)
                                labels.append(label)
                                if save_segments:
                                    self._save_segment(clip, segments_dir, root, file, i, sr)
                    else:
                        tqdm.write(f"⚠️ No valid segments found in {file}")
                pbar.update(1)

        print(f"\n✅ Audio processing complete!")
        print(f" 📁 Files found: {len(all_files)}")
        print(f" ✅ Processed: {stats['processed']}")
        print(f" ❌ Failed: {stats['failed']}")
        print(f" 🎵 Total segments: {stats['segments']}")
        if self.use_quality_filter:
            print(f" 🔍 Quality filtering was enabled - cleaner segments should result in better spectrograms")
        return audio_clips, labels

    def _discover_segments(self, segments_dir):
        """Discover existing segments, applying dev mode limits if enabled."""
        all_files = []
        species_counts = {} if self.dev_mode else None
        for root, _, files in os.walk(segments_dir):
            wav_files = [f for f in files if f.endswith('.wav')]
            for file in wav_files:
                file_path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                if self.dev_mode:
                    if label not in species_counts:
                        species_counts[label] = 0
                    if species_counts[label] >= self.dev_limit:
                        continue
                    species_counts[label] += 1
                all_files.append((file_path, file, label))
        return all_files

    def load_existing_segments(self, segments_dir=None):
        """Load existing audio segments from the segments directory."""
        segments_dir = segments_dir or self.segments_dir
        audio_clips, labels = [], []
        all_files = self._discover_segments(segments_dir)
        if not all_files:
            print(f"No segments found in {segments_dir}.")
            return audio_clips, labels

        if self.dev_mode:
            print(f"🔧 DEV MODE: Limited to {self.dev_limit} segments per species")
            print(f"Will load {len(all_files)} segments from {len(set([lbl for _, _, lbl in all_files]))} species")

        stats = {"loaded": 0, "failed": 0, "rejected": 0}
        sr = self.config['target_sr']
        target_length = int(self.duration * sr)

        with tqdm(total=len(all_files), desc="Loading segments", unit="segment") as pbar:
            for file_path, file, label in all_files:
                pbar.set_postfix_str(f"Species: {label}")
                audio, error = self.load_and_validate_audio(file_path, target_sr=sr)
                if error:
                    tqdm.write(f"❌ Failed to load {file}: {error}")
                    stats["failed"] += 1
                else:
                    audio = self.pad_or_trim_audio(audio, target_length)
                    if self.use_quality_filter:
                        quality_result = self.assess_audio_quality(audio, sr)
                        if not quality_result["pass"]:
                            stats["rejected"] += 1
                            pbar.update(1)
                            continue
                    audio_clips.append(audio)
                    labels.append(label)
                    stats["loaded"] += 1
                pbar.update(1)

        print(f"\n✅ Loading complete!")
        print(f" 📊 Total available: {len(all_files)}")
        print(f" ✅ Loaded: {stats['loaded']}")
        print(f" ❌ Failed: {stats['failed']}")
        if self.use_quality_filter:
            print(f" 🔍 Quality rejected: {stats['rejected']}")
            total_checked = stats['loaded'] + stats['rejected']
            pass_rate = (stats['loaded'] / total_checked * 100) if total_checked > 0 else 0
            print(f" 📈 Quality pass rate: {pass_rate:.1f}%")
        return audio_clips, labels

    def resample_audio(self, download_dir="downloads", resampled_dir=None):
        """Resample audio files to target SR and organize them."""
        resampled_dir = resampled_dir or self.data_dir
        if input("Do you want to resample audio files? (y/n): ").strip().lower() != 'y':
            print("Skipping audio resampling.")
            return

        if os.path.exists(resampled_dir):
            print(f"Directory {resampled_dir} already exists. Deleting entire folder.")
            shutil.rmtree(resampled_dir)
        os.makedirs(resampled_dir)

        all_filenames = [f for f in os.listdir(download_dir) if f.endswith(".wav")]
        if self.dev_mode:
            print(f"🔧 DEV MODE: Limiting resampling to {self.dev_limit} files per species")
            species_file_counts = {}
            filenames = []
            for filename in all_filenames:
                try:
                    parts = filename[:-4].split('_')
                    if len(parts) >= 4:
                        english_name = parts[1]
                        if english_name not in species_file_counts:
                            species_file_counts[english_name] = 0
                        if species_file_counts[english_name] < self.dev_limit:
                            filenames.append(filename)
                            species_file_counts[english_name] += 1
                except:
                    continue
            print(f"Selected {len(filenames)} files from {len(species_file_counts)} species for resampling")
        else:
            filenames = all_filenames

        stats = {"processed": 0, "skipped": 0, "failed": 0}
        sr = self.config['target_sr']

        with tqdm(total=len(filenames), desc="Processing audio files", unit="file") as pbar:
            for filename in filenames:
                path = os.path.join(download_dir, filename)
                pbar.set_postfix_str(f"Current: {filename[:30]}...")
                try:
                    parts = filename[:-4].split('_')
                    if len(parts) < 4:
                        tqdm.write(f"❌ Filename format not recognized: {filename}")
                        stats["failed"] += 1
                        pbar.update(1)
                        continue
                    file_id = parts[0]
                    english_name = parts[1]
                    genus = parts[2]
                    species = parts[3]

                    if os.path.getsize(path) == 0:
                        tqdm.write(f"⚠️ Skipping empty file: {filename}")
                        stats["skipped"] += 1
                        pbar.update(1)
                        continue

                    audio, error = self.load_and_validate_audio(path, min_duration=0.5, target_sr=None)
                    if error:
                        tqdm.write(f"❌ Failed to load {filename}: {error}")
                        stats["failed"] += 1
                        pbar.update(1)
                        continue

                    sr_loaded = librosa.get_samplerate(path)
                    if sr_loaded < sr:
                        tqdm.write(f"⚠️ Skipping {filename}: Sample rate {sr_loaded} too low")
                        stats["skipped"] += 1
                        pbar.update(1)
                        continue

                    if self.use_noise_reduction:
                        audio = self.apply_noise_reduction(audio, sr)

                    nested_folder = os.path.join(resampled_dir, english_name)
                    os.makedirs(nested_folder, exist_ok=True)
                    out_path = os.path.join(nested_folder, filename)
                    sf.write(out_path, audio, sr)
                    stats["processed"] += 1
                except Exception as e:
                    tqdm.write(f"❌ Failed to process {filename}: {str(e)}")
                    stats["failed"] += 1
                pbar.update(1)

        print(f"\n✅ Resampling complete!")
        print(f" 📊 Processed: {stats['processed']}")
        print(f" ⚠️ Skipped: {stats['skipped']}")
        print(f" ❌ Failed: {stats['failed']}")
        print(f" 📁 Total: {len(filenames)}")

def main():
    """Main function to process bird sound data and prepare for ViT training."""
    print("🐦 Bird Sound Classification with Vision Transformer")
    print("=" * 55)
    dev_mode = input("Enable dev mode? (10 segments per species for testing) (y/n): ").strip().lower() == 'y'
    use_quality_filter = input("Enable advanced quality filtering? (removes noisy/poor segments) (y/n): ").strip().lower() == 'y'
    use_noise_reduction = input("Apply noise reduction to audio? (removes background noise and artifacts) (y/n): ").strip().lower() == 'y'

    if dev_mode:
        print("🔧 DEV MODE: Loading limited data for quick testing")
    processor = AudioProcessor(
        dev_mode=dev_mode,
        dev_limit=10,
        duration=4.0,
        use_quality_filter=use_quality_filter,
        use_noise_reduction=use_noise_reduction
    )
    print()
    print("📁 Loading audio data...")
    audio_clips, labels = processor.load_audio_data()
    print(f"✅ Loaded {len(audio_clips)} audio clips")

    print("🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(set(encoded_labels))
    print(f"✅ Encoded {num_classes} unique species")

    print("\n🎯 Data preparation complete! Ready for ViT model training.")
    print(f" 🐦 Classes: {num_classes}")
    print(f" 📈 Training samples: {len(audio_clips)}")

    if dev_mode:
        print(f"\n🔧 DEV MODE SUMMARY:")
        print(f" • Limited to {processor.dev_limit} files per species during resampling")
        print(f" • Limited to {processor.dev_limit} files per species during segmentation")
        print(f" This ensures faster testing with representative data from each species.")
    if use_noise_reduction:
        print(f" • Noise reduction was applied during preprocessing")
    if use_quality_filter:
        print(f" • Quality filtering was applied")

if __name__ == "__main__":
    main()