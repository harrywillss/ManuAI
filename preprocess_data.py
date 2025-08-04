'''
Audio processing module for bird sound classification.
This module handles audio loading, segmentation, feature extraction, and data preparation for training.
'''

import os
import shutil

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import io
from scipy.signal import medfilt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

os.nice(10)  # makes the process "nicer" to your system

class AudioProcessor:
    """Handles audio processing, feature extraction, and quality assessment for bird sound classification."""
    
    def __init__(self, data_dir="training_data", duration=4, save_segments=True, 
                 segments_dir="segments", dev_mode=False, dev_limit=10):
        self.data_dir = data_dir
        self.duration = duration
        self.save_segments = save_segments
        self.segments_dir = segments_dir
        self.dev_mode = dev_mode
        self.dev_limit = dev_limit
        
        # State variables for user preferences
        self._asked_save_spectrograms = False
        self._asked_quality_filter = False
        self._asked_noise_reduction = False
        self.save_spectrograms_enabled = False
        self.use_quality_filter = False
        self.use_noise_reduction = False

    def _ask_user_preference(self, question, default=None):
        """Helper to ask user yes/no questions."""
        if default is not None:
            return default
        response = input(f"{question} (y/n): ").strip().lower()
        return response == 'y'

    def ask_save_spectrograms_preference(self):
        """Ask user if they want to save spectrograms during processing."""
        if not self._asked_save_spectrograms:
            self.save_spectrograms_enabled = self._ask_user_preference(
                "Save spectrograms in ed folders? (creates spectrograms/ directory)"
            )
            self._asked_save_spectrograms = True
            
            status = "enabled" if self.save_spectrograms_enabled else "disabled"
            print(f"📊 Spectrogram saving {status}")
            
        return self.save_spectrograms_enabled

    def ask_noise_reduction_preference(self):
        """Ask user if they want to apply noise reduction during processing."""
        if not self._asked_noise_reduction:
            self.use_noise_reduction = self._ask_user_preference(
                "Apply noise reduction to audio? (removes background noise and artifacts)"
            )
            self._asked_noise_reduction = True
            
            status = "enabled" if self.use_noise_reduction else "disabled"
            print(f"🔧 Noise reduction {status}")
            
        return self.use_noise_reduction

    def reduce_noise_spectral_gating(self, audio, sr, noise_gate_threshold=0.02, 
                                   noise_reduce_factor=0.5):
        """
        Apply noise reduction using spectral gating technique.
        This method estimates noise from quiet sections and reduces it throughout.
        """
        try:
            # Convert to STFT for frequency domain processing
            stft = librosa.stft(audio, hop_length=256, win_length=1024)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor from quieter sections
            # Use bottom 10% of magnitudes as noise estimate
            sorted_mags = np.sort(magnitude.flatten())
            noise_floor = np.mean(sorted_mags[:int(len(sorted_mags) * 0.1)])
            
            # Apply spectral gating
            noise_mask = magnitude < (noise_gate_threshold * np.max(magnitude))
            
            # Reduce noise in identified sections
            reduced_magnitude = magnitude.copy()
            reduced_magnitude[noise_mask] *= noise_reduce_factor
            
            # Reconstruct audio
            reduced_stft = reduced_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(reduced_stft, hop_length=256, win_length=1024)
            
            return cleaned_audio
            
        except Exception as e:
            print(f"Warning: Noise reduction failed, using original audio: {e}")
            return audio

    def reduce_noise_median_filter(self, audio, kernel_size=3):
        """
        Apply median filtering to reduce impulsive noise and clicks.
        Good for removing sudden spikes and clicks in audio.
        """
        try:
            # Apply median filter to reduce impulsive noise
            filtered_audio = medfilt(audio, kernel_size=kernel_size)
            return filtered_audio
        except Exception as e:
            print(f"Warning: Median filtering failed, using original audio: {e}")
            return audio

    def apply_noise_reduction(self, audio, sr):
        """
        Apply comprehensive noise reduction pipeline.
        Combines multiple techniques for better results.
        """
        if not self.use_noise_reduction:
            return audio
            
        try:
            # Step 1: Remove DC offset
            audio = audio - np.mean(audio)
            
            # Step 2: Apply median filter for click/pop removal
            audio = self.reduce_noise_median_filter(audio, kernel_size=3)
            
            # Step 3: Apply spectral gating for background noise
            audio = self.reduce_noise_spectral_gating(audio, sr, 
                                                    noise_gate_threshold=0.02,
                                                    noise_reduce_factor=0.6)
            
            # Step 4: Normalize to prevent clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            return audio
            
        except Exception as e:
            print(f"Warning: Noise reduction pipeline failed, using original audio: {e}")
            return audio

    def assess_audio_quality(self, audio, sr, min_snr=10.0, max_silence_ratio=0.7, min_spectral_centroid=500, max_spectral_centroid=8000):
        """
        Assess the quality of an audio clip based on SNR, silence, and spectral characteristics.
        """
        try:
            # 1. Signal-to-Noise Ratio (SNR) estimation
            # Use top 10% vs bottom 10% of RMS energy as proxy for signal vs noise
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            if len(rms) < 10:  # Too short for analysis
                return {"quality_score": 0, "pass": False, "reason": "Too short for analysis"}
            
            # Sort RMS values to estimate signal vs noise
            rms_sorted = np.sort(rms)
            noise_level = np.mean(rms_sorted[:len(rms_sorted)//10])  # Bottom 10%
            signal_level = np.mean(rms_sorted[-len(rms_sorted)//10:])  # Top 10%
            
            snr_db = 20 * np.log10(signal_level / max(noise_level, 1e-10))
            
            # 2. Silence detection
            # Count frames below a dynamic threshold
            threshold = np.mean(rms) * 0.1
            silence_frames = np.sum(rms < threshold)
            silence_ratio = silence_frames / len(rms)
            
            # 3. Spectral characteristics
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            avg_spectral_centroid = np.mean(spectral_centroids)
            
            # 4. Zero crossing rate (helps detect noise/clicks)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            avg_zcr = np.mean(zcr)
            
            # 5. Spectral rolloff (frequency below which 85% of energy is contained)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            avg_rolloff = np.mean(spectral_rolloff)
            
            # Quality scoring
            quality_score = 0
            reasons = []
            
            # SNR check
            if snr_db >= min_snr:
                quality_score += 30
            else:
                reasons.append(f"Low SNR: {snr_db:.1f}dB < {min_snr}dB")
            
            # Silence check
            if silence_ratio <= max_silence_ratio:
                quality_score += 25
            else:
                reasons.append(f"Too much silence: {silence_ratio:.1%} > {max_silence_ratio:.1%}")
            
            # Spectral centroid check (bird calls typically 1-6kHz)
            if min_spectral_centroid <= avg_spectral_centroid <= max_spectral_centroid:
                quality_score += 25
            else:
                reasons.append(f"Spectral centroid out of range: {avg_spectral_centroid:.0f}Hz")
            
            # Zero crossing rate check (too high suggests noise/clicks)
            if avg_zcr < 0.3:  # Reasonable threshold for bird calls
                quality_score += 20
            else:
                reasons.append(f"High zero crossing rate: {avg_zcr:.3f} (possible noise)")
            
            # Overall assessment
            quality_pass = quality_score >= 60  # Need at least 60/100 to pass
            
            return {
                "quality_score": quality_score,
                "pass": quality_pass,
                "snr_db": snr_db,
                "silence_ratio": silence_ratio,
                "spectral_centroid": avg_spectral_centroid,
                "zero_crossing_rate": avg_zcr,
                "spectral_rolloff": avg_rolloff,
                "reasons": reasons
            }
            
        except Exception as e:
            return {"quality_score": 0, "pass": False, "reason": f"Analysis failed: {str(e)}"}
    
    def segment_audio_snr(self, audio, sr, segment_len_ms=2500, hop_len_ms=1000, noise_len_ms=500, snr_threshold=1.0):
        """
        Segment audio based on SNR (Signal-to-Noise Ratio) using librosa.
        This function detects onsets and extracts segments of a specified length around those onsets.
        """

        # Convert lengths from milliseconds to samples
        segment_len_samples = int(sr * segment_len_ms / 1000)
        hop_len_samples = int(sr * hop_len_ms / 1000)
        noise_len_samples = int(sr * noise_len_ms / 1000)

        def get_noise_level(samples):
            """
            Calculate the noise level from short segments of audio.
            """
            abs_max = []
            if len(samples) > noise_len_samples:
                for i in range(0, len(samples) - noise_len_samples, noise_len_samples):
                    abs_max.append(np.max(np.abs(samples[i:i + noise_len_samples])))
            else:
                abs_max.append(np.max(np.abs(samples)))
            return min(abs_max) if abs_max else 1e-6

        # If audio is too short, return empty list
        if len(audio) < segment_len_samples:
            return []

        noise_level = get_noise_level(audio)
        
        try:
            # Try onset detection
            onsets = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=hop_len_samples, backtrack=True)
            onset_samples = librosa.frames_to_samples(onsets, hop_length=hop_len_samples)
            segments = []
            
            for onset in onset_samples:
                start = max(0, onset - segment_len_samples // 4)  # Center segment around onset
                end = start + segment_len_samples
                if end <= len(audio):
                    segment = audio[start:end]
                    if len(segment) == segment_len_samples: 
                        segments.append(segment) # Only add if segment is full length
        except Exception as e:
            # If onset detection fails, return empty list
            print(f"Onset detection failed: {e}")
            return []

        if not segments:
            # If no segments found, try sliding window approach
            for i in range(0, len(audio) - segment_len_samples, hop_len_samples):
                segment = audio[i:i + segment_len_samples]
                if len(segment) < segment_len_samples * 0.5:  # Skip very short segments
                    continue
                seg_abs_max = np.max(np.abs(segment))
                snr = seg_abs_max / noise_level if noise_level != 0 else 0 # Calculate SNR 
                if snr > snr_threshold:
                    if len(segment) == segment_len_samples:
                        segments.append(segment) # Only add if segment is full length

        return segments

    def load_audio_data(self, data_dir="training_data", save_segments=True, segments_dir="segments"):
        """
        Load audio data from the specified directory, segment it, and return audio clips and labels.
        """
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist.")
            self.resample_audio()
        else:
            print(f"Directory {data_dir} already exists. Skipping resampling.")

        # Check if user wants to skip segmentation and use existing segments
        if os.path.exists(segments_dir) and input("Segments directory exists. Do you want to skip segmentation and load existing segments? (y/n): ").strip().lower() == 'y':
            print("Loading existing segments...")
            return self.load_existing_segments(segments_dir, self.duration)

        audio_clips = []
        labels = []
        total_files = 0
        processed_files = 0
        failed_files = 0

        # Create segments directory if saving segments
        if save_segments and not os.path.exists(segments_dir):
            os.makedirs(segments_dir)
        
        # First pass: count total files and apply dev mode limiting
        all_files = []
        species_file_counts = {} if self.dev_mode else None
        
        for root, dirs, files in os.walk(data_dir):
            wav_files = [f for f in files if f.endswith('.wav')]
            for file in wav_files:
                # Use the English name as the label (parent folder of scientific name)
                label = os.path.basename(os.path.dirname(root))
                
                # Dev mode: limit files per species during initial loading
                if self.dev_mode:
                    if label not in species_file_counts:
                        species_file_counts[label] = 0
                    if species_file_counts[label] >= self.dev_limit:
                        continue  # Skip this file, already have enough for this species
                    species_file_counts[label] += 1
                
                all_files.append((root, file))
        
        total_files = len(all_files)
        if total_files == 0:
            print(f"No audio files found in {data_dir}. Please check the directory.")
            return audio_clips, labels

        if self.dev_mode:
            print(f"🔧 DEV MODE: Limited to {self.dev_limit} files per species during segmentation")
            print(f"Will process {total_files} files from {len(species_file_counts)} species")

        # Process files with progress bar
        with tqdm(total=total_files, desc="Processing audio files", unit="file") as pbar:
            for root, file in all_files:
                file_path = os.path.join(root, file)
                # Use the English name as the label (parent folder of scientific name)
                label = os.path.basename(os.path.dirname(root))
                
                pbar.set_postfix_str(f"Current: {file[:25]}... | Species: {label}")
                
                try:
                    # Load the audio file
                    audio, sr = librosa.load(file_path, sr=None)
                    
                    # Skip very short files
                    if len(audio) < sr * 1.0:  # Less than 1 second
                        tqdm.write(f"⚠️  Skipping {file}: Audio too short ({len(audio)/sr:.2f}s)")
                        continue
                    
                    # Use simple fixed-duration chunking with overlap
                    segments = []
                    chunk_size = int(self.duration * sr)
                    hop_size = chunk_size // 2  # 50% overlap
                    
                    # Ask user if they want quality filtering (only ask once)
                    if not hasattr(self, '_asked_quality_filter'):
                        self.use_quality_filter = input("Enable advanced quality filtering? (removes noisy/poor segments) (y/n): ").strip().lower() == 'y'
                        self._asked_quality_filter = True
                        if self.use_quality_filter:
                            print("🔍 Quality filtering enabled - will assess SNR, silence, and spectral characteristics")
                    
                    quality_stats = {"total": 0, "passed": 0, "failed": 0}
                    
                    for start in range(0, len(audio) - chunk_size + 1, hop_size):
                        chunk = audio[start:start + chunk_size]
                        if len(chunk) == chunk_size:
                            quality_stats["total"] += 1
                            
                            # Basic amplitude check first (quick filter)
                            if np.max(np.abs(chunk)) <= 0.01:  # Too quiet
                                quality_stats["failed"] += 1
                                continue
                            
                            # Advanced quality assessment if enabled
                            if self.use_quality_filter:
                                quality_result = self.assess_audio_quality(chunk, sr)
                                if quality_result["pass"]:
                                    segments.append(chunk) # Only add if quality passes
                                    quality_stats["passed"] += 1
                                else:
                                    quality_stats["failed"] += 1
                                    # Optionally log why it failed (for debugging)
                                    # if len(quality_result.get("reasons", [])) > 0:
                                    #     tqdm.write(f"   ⚠️ Rejected segment: {', '.join(quality_result['reasons'][:2])}")
                            else:
                                # If quality filter is off, just add the chunk
                                segments.append(chunk)
                                quality_stats["passed"] += 1
                    
                    # Update progress bar with quality stats
                    if quality_stats["total"] > 0:
                        pass_rate = quality_stats["passed"] / quality_stats["total"] * 100
                        pbar.set_postfix_str(f"Species: {label} | Quality: {pass_rate:.0f}% pass rate")
                    
                    # If still no segments, try the SNR-based method as fallback
                    if len(segments) == 0:
                        try:
                            snr_segments = self.segment_audio_snr(audio, sr, segment_len_ms=3000, snr_threshold=1.0)
                            # Apply quality filter to SNR segments too if enabled
                            if self.use_quality_filter:
                                for seg in snr_segments:
                                    quality_result = self.assess_audio_quality(seg, sr)
                                    if quality_result["pass"]:
                                        segments.append(seg)  # Only add if quality passes
                            else:
                                segments.extend(snr_segments)
                        except:
                            # Final fallback - just take the middle portion
                            if len(audio) >= chunk_size:
                                start = (len(audio) - chunk_size) // 2
                                chunk = audio[start:start + chunk_size]
                                # Apply quality check to fallback segment too
                                if np.max(np.abs(chunk)) > 0.01:
                                    if self.use_quality_filter:
                                        quality_result = self.assess_audio_quality(chunk, sr)
                                        if quality_result["pass"]:
                                            segments.append(chunk)  # Only add if quality passes
                                    else:
                                        segments.append(chunk)
                    
                    segments_added = 0
                    for i, clip in enumerate(segments):
                        expected_length = self.duration * sr
                        if len(clip) >= expected_length * 0.75:  # Accept segments that are at least 75% of the expected length
                            # Pad or trim to exact duration
                            if len(clip) < expected_length:
                                clip = np.pad(clip, (0, expected_length - len(clip)), mode='constant')
                            elif len(clip) > expected_length:
                                clip = clip[:expected_length]
                            
                            audio_clips.append(clip)
                            labels.append(label)
                            segments_added += 1
                            
                            # Save segment if requested
                            if save_segments:
                                # Create nested structure in segments folder
                                english_name = os.path.basename(os.path.dirname(root))
                                scientific_name = os.path.basename(root)
                                segment_folder = os.path.join(segments_dir, english_name, scientific_name)
                                os.makedirs(segment_folder, exist_ok=True)
                                
                                # Save segment with original filename + segment number
                                segment_filename = f"{file[:-4]}_segment_{i}.wav"
                                segment_path = os.path.join(segment_folder, segment_filename)
                                sf.write(segment_path, clip, sr)
                    if segments_added > 0:
                        processed_files += 1
                    else:
                        tqdm.write(f"⚠️  No valid segments found in {file}")
                    
                except Exception as e:
                    tqdm.write(f"❌ Failed to process {file}: {str(e)}")
                    failed_files += 1
                    
                pbar.update(1)

        print(f"\n✅ Audio processing complete!")
        print(f"   📁 Files found: {total_files}")
        print(f"   ✅ Processed: {processed_files}")
        print(f"   ❌ Failed: {failed_files}")
        print(f"   🎵 Total segments: {len(audio_clips)}")
        
        # Show quality filtering summary if it was used
        if hasattr(self, 'use_quality_filter') and self.use_quality_filter:
            print(f"   🔍 Quality filtering was enabled - cleaner segments should result in better spectrograms")
        
        return audio_clips, labels

    def load_existing_segments(self, segments_dir):
        """
        Load existing audio segments from the segments directory.
        
        Args:
            segments_dir: Directory containing audio segments
            duration: Expected duration of segments in seconds
        """
        audio_clips = []
        labels = []
        
        print(f"Scanning segments directory: {segments_dir}")
        
        # Collect all file paths first
        all_files = []
        species_counts = {} if self.dev_mode else None
        
        for root, dirs, files in os.walk(segments_dir):
            wav_files = [f for f in files if f.endswith('.wav')]
            for file in wav_files:
                file_path = os.path.join(root, file)
                # Use the English name as the label (grandparent folder of segment file)
                # Structure: segments_dir/english_name/scientific_name/segment_file.wav
                label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                
                # Dev mode: limit segments per species
                if self.dev_mode:
                    if label not in species_counts:
                        species_counts[label] = 0
                    if species_counts[label] >= self.dev_limit:
                        continue  # Skip this file, already have enough for this species
                    species_counts[label] += 1
                
        
        total_segments = len(all_files)
        
        if self.dev_mode:
            print(f"🔧 DEV MODE: Limited to {self.dev_limit} segments per species")
            print(f"Will load {total_segments} segments from {len(species_counts)} species")
        else:
            print(f"Found {total_segments} total segments to load")
        
        # Ask about quality filtering
        use_quality_filter = input("Apply quality filtering to existing segments? (may remove poor quality segments) (y/n): ").strip().lower() == 'y'
        if use_quality_filter:
            print("🔍 Quality filtering enabled for existing segments")
        
        loaded_segments = 0
        failed_segments = 0
        quality_rejected = 0
        
        # Load segments with progress bar
        with tqdm(total=len(all_files), desc="Loading segments", unit="segment") as pbar:
            for file_path, file, label in all_files:
                pbar.set_postfix_str(f"Species: {label}")
                
                try:
                    # Load the audio segment
                    audio, sr = librosa.load(file_path, sr=None)
                    expected_length = int(self.duration * sr)
                    
                    # Ensure the segment has the correct duration
                    if len(audio) < expected_length:
                        # Pad if too short
                        audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
                    elif len(audio) > expected_length:
                        # Trim if too long
                        audio = audio[:expected_length]
                    
                    # Apply quality filtering if requested
                    if use_quality_filter:
                        quality_result = self.assess_audio_quality(audio, sr)
                        if not quality_result["pass"]:
                            quality_rejected += 1
                            pbar.update(1)
                            continue
                    
                    audio_clips.append(audio)
                    labels.append(label)
                    loaded_segments += 1
                    
                except Exception as e:
                    tqdm.write(f"❌ Failed to load {file}: {str(e)}")
                    failed_segments += 1
                
                pbar.update(1)
        
        # Final reporting
        if self.dev_mode:
            final_counts = {}
            for label in labels:
                final_counts[label] = final_counts.get(label, 0) + 1
            print(f"🔧 DEV MODE complete! Loaded {loaded_segments} segments from {len(final_counts)} species")
            print(f"   Species distribution: {dict(final_counts)}")
        
        print(f"\n✅ Loading complete!")
        print(f"   📊 Total available: {total_segments}")
        print(f"   ✅ Loaded: {loaded_segments}")
        print(f"   ❌ Failed: {failed_segments}")
        if use_quality_filter:
            print(f"   🔍 Quality rejected: {quality_rejected}")
            print(f"   📈 Quality pass rate: {(loaded_segments/(loaded_segments + quality_rejected)*100):.1f}%" if (loaded_segments + quality_rejected) > 0 else "   📈 Quality pass rate: N/A")
        
        return audio_clips, labels
    
    def validate_wav_file(self, file_path):
        try:
            audio, sr = sf.read(file_path)
            if audio.size == 0:
                print(f"Warning: {file_path} is empty")
                return False
            return True
        except Exception as e:
            print(f"Validation failed for {file_path}: {str(e)}")
            return False

    def extract_features(self, audio_files, target_width=224):
        """
        Extract mel-spectrogram features from audio files.
        
        Args:
            audio_files: List of audio arrays
            target_width: Target width for spectrograms (224 for ViT compatibility)
        """
        features = []
        failed_count = 0
        
        print(f"Extracting mel-spectrogram features from {len(audio_files)} audio clips...")
        
        with tqdm(total=len(audio_files), desc="Extracting features", unit="clip") as pbar:
            for audio in audio_files:
                try:
                    # Normalize audio
                    if np.max(np.abs(audio)) != 0:
                        audio = audio / np.max(np.abs(audio))
                    
                    # Extract mel-spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio, sr=44100, n_mels=224, fmax=8000,
                        hop_length=256, win_length=1024
                    )
                    
                    # Convert to dB and adjust width
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    mel_spec_db = self._adjust_spectrogram_width(mel_spec_db, target_width)
                    
                    features.append(mel_spec_db)
                    pbar.set_postfix_str(f"Shape: {mel_spec_db.shape}")
                except Exception:
                    failed_count += 1
                
                pbar.update(1)
        
        success_count = len(features)
        print(f"✅ Extracted {success_count} feature matrices" + 
              (f", failed: {failed_count}" if failed_count > 0 else ""))
        
        return features
    
    def _adjust_spectrogram_width(self, mel_spec_db, target_width):
        """Adjust spectrogram width by padding or trimming."""
        if mel_spec_db.shape[1] < target_width:
            pad_width = target_width - mel_spec_db.shape[1]
            return np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_spec_db.shape[1] > target_width:
            return mel_spec_db[:, :target_width]
        return mel_spec_db
    
    def save_spectrograms(self, mel_features, labels, output_dir="spectrograms"):
        """Save mel-spectrograms with multiple storage options."""
        print("\n📊 Spectrogram Storage Options:")
        print("1. Skip saving (recommended - generate on-demand during training)")
        print("2. Save as compressed numpy array (.npz)")
        print("3. Save as individual PNG files (not recommended for large datasets)")
        
        choice = input("Choose option (1/2/3): ").strip()
        
        if choice == "1":
            print("✅ Skipping spectrogram saving - will generate on-demand during training")
            return True
        
        elif choice == "2":
            return self._save_spectrograms_compressed(mel_features, labels, output_dir)
        
        elif choice == "3":
            if len(mel_features) > 5000:
                confirm = input(f"⚠️  Saving {len(mel_features)} PNG files will use significant disk space. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    print("Cancelled PNG export")
                    return False
            return self._save_spectrograms_png(mel_features, labels, output_dir)
        
        else:
            print("Invalid choice, skipping spectrogram saving")
            return False

    def _save_spectrograms_compressed(self, mel_features, labels, output_dir):
        """Save spectrograms in compressed numpy format."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "spectrograms_compressed.npz")
            
            # Apply dev mode filtering if needed
            if self.dev_mode:
                mel_features, labels = self._apply_dev_mode_filtering(mel_features, labels)
            
            print(f"💾 Saving {len(mel_features)} spectrograms to compressed format...")
            
            # Save with metadata
            np.savez_compressed(
                output_file,
                spectrograms=np.array(mel_features),
                labels=np.array(labels),
                metadata=np.array([{
                    'num_samples': len(mel_features),
                    'num_classes': len(set(labels)),
                    'shape': np.array(mel_features).shape,
                    'dev_mode': self.dev_mode,
                    'dev_limit': self.dev_limit if self.dev_mode else None
                }], dtype=object)
            )
            
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ Spectrograms saved successfully!")
            print(f"   📁 File: {output_file}")
            print(f"   💾 Size: {file_size_mb:.1f} MB")
            print(f"   🎵 Samples: {len(mel_features)}")
            print(f"   🐦 Species: {len(set(labels))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving compressed spectrograms: {str(e)}")
            return False

    def _save_spectrograms_png(self, mel_features, labels, output_dir):
        """Save spectrograms as PNG files (original implementation)."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Apply dev mode filtering if needed
            if self.dev_mode:
                mel_features, labels = self._apply_dev_mode_filtering(mel_features, labels)
            
            species_counts = {}
            
            with tqdm(total=len(mel_features), desc="Saving spectrograms", unit="file") as pbar:
                for i, (mel, label) in enumerate(zip(mel_features, labels)):
                    try:
                        # Track count for this species
                        species_counts[label] = species_counts.get(label, 0) + 1
                        
                        # Create species folder and filename
                        species_folder = os.path.join(output_dir, label)
                        os.makedirs(species_folder, exist_ok=True)
                        filename = f"{label}_spectrogram_{species_counts[label]:04d}.png"
                        filepath = os.path.join(species_folder, filename)
                        
                        # Create and save spectrogram with ViT-compatible settings
                        plt.figure(figsize=(6, 6))
                        plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
                        plt.axis('off')
                        plt.tight_layout()
                        
                        # Save to memory buffer first
                        buf = io.BytesIO()
                        plt.savefig(buf, dpi=75, bbox_inches='tight', 
                                  pad_inches=0, facecolor='white', format='png')
                        plt.close()
                        
                        # Load from buffer and process for ViT compatibility
                        buf.seek(0)
                        img = Image.open(buf)
                        
                        # Convert RGBA to RGB (remove alpha channel)
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        
                        # Resize to 224x224 for ViT compatibility
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)
                        
                        # Save final image
                        img.save(filepath, format='PNG')
                        buf.close()
                        
                        pbar.set_postfix_str(f"Species: {label}")
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"❌ Error saving spectrogram {i} ({label}): {str(e)}")
                        pbar.update(1)
            
            print(f"✅ Spectrograms saved successfully!")
            print(f"   📁 Output: {output_dir}")
            print(f"   🎵 Total: {len(mel_features)}")
            if self.dev_mode:
                print(f"   🔧 DEV MODE: Limited to {self.dev_limit} per species")
            print(f"   🐦 Species distribution:")
            for species, count in sorted(species_counts.items()):
                print(f"      {species}: {count}")
                
            return True
                
        except Exception as e:
            print(f"❌ Error in save_spectrograms: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _apply_dev_mode_filtering(self, mel_features, labels):
        """Apply dev mode filtering to reduce dataset size."""
        if not self.dev_mode:
            return mel_features, labels
        
        print(f"🔧 DEV MODE: Filtering to max {self.dev_limit} per species...")
        species_counts = {}
        filtered_features = []
        filtered_labels = []
        
        for mel, label in zip(mel_features, labels):
            if label not in species_counts:
                species_counts[label] = 0
            if species_counts[label] < self.dev_limit:
                filtered_features.append(mel)
                filtered_labels.append(label)
                species_counts[label] += 1
        
        print(f"Filtered to {len(filtered_features)} spectrograms from {len(species_counts)} species")
        return filtered_features, filtered_labels

    def generate_spectrogram_on_demand(self, audio_segment, target_size=(224, 224)):
        """Generate spectrogram on-demand without saving to disk."""
        try:
            # Normalize audio
            if np.max(np.abs(audio_segment)) != 0:
                audio_segment = audio_segment / np.max(np.abs(audio_segment))
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment, sr=44100, n_mels=target_size[0], fmax=8000,
                hop_length=256, win_length=1024
            )
            
            # Convert to dB and adjust width
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = self._adjust_spectrogram_width(mel_spec_db, target_size[1])
            
            # Convert to PIL Image for ViT compatibility
            plt.figure(figsize=(6, 6))
            plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
            plt.axis('off')
            plt.tight_layout()
            
            # Save to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, dpi=75, bbox_inches='tight', 
                      pad_inches=0, facecolor='white', format='png')
            plt.close()
            
            # Load from buffer and process for ViT compatibility
            buf.seek(0)
            img = Image.open(buf)
            
            # Convert RGBA to RGB (remove alpha channel)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Resize to target size for ViT compatibility
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            buf.close()
            
            return img
            
        except Exception as e:
            print(f"❌ Error generating spectrogram: {str(e)}")
            return None

    def create_image_patches_with_position(
        self, features, labels, patch_size=16, height=224, width=224, expected_sequence_length=196, batch_size=50
    ):
        """
        Creates image patches with positional encodings using memory-efficient batching.
        For ViT: 224x224 images → 16x16 patches = 196 patches × 258 features each (256 + 2 pos encoding)
        """
        # Dev mode: reduce batch size for faster processing with limited data
        if self.dev_mode:
            batch_size = min(batch_size, 20)  # Smaller batches for dev mode
            print(f"🧠 Creating {len(features)} image patches (DEV MODE: batch_size={batch_size})")
        else:
            print(f"🧠 Creating {len(features)} image patches in batches of {batch_size} (memory-efficient)")
        
        # Calculate dimensions and memory estimates
        grid_height = height // patch_size
        grid_width = width // patch_size
        patch_features = patch_size * patch_size + 2  # 256 pixel values + 2 positional encodings
        
        # Estimate memory per batch and adjust if needed
        estimated_gb_per_batch = (batch_size * expected_sequence_length * patch_features * 8) / (1024**3)
        if estimated_gb_per_batch > 2.0:  # Limit to 2GB per batch
            batch_size = max(10, int(batch_size * 2.0 / estimated_gb_per_batch))
            batch_size_msg = "memory safety"
            if self.dev_mode:
                batch_size_msg += " (DEV MODE)"
            print(f"⚠️  Reduced batch size to {batch_size} for {batch_size_msg}")

        # Create positional encodings once
        pos_enc_height = np.repeat(np.arange(grid_height)[:, np.newaxis], grid_width, axis=1)
        pos_enc_width = np.repeat(np.arange(grid_width)[np.newaxis, :], grid_height, axis=0)

        all_sequences = []
        total_batches = (len(features) + batch_size - 1) // batch_size
        
        with tqdm(total=len(features), desc="Creating patches", unit="sample") as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(features))
                batch_features = features[start_idx:end_idx]
                
                batch_sequences = []
                
                for feature in batch_features:
                    # Ensure correct dimensions (pad/trim if needed)
                    feature = self._resize_feature(feature, height, width)
                    
                    # Create patches for this feature
                    current_sequence = []
                    for i in range(0, height, patch_size):
                        for j in range(0, width, patch_size):
                            patch = feature[i:i + patch_size, j:j + patch_size]
                            
                            # Add positional encoding
                            pos_enc = np.array([
                                pos_enc_height[i // patch_size, j // patch_size],
                                pos_enc_width[i // patch_size, j // patch_size]
                            ])
                            
                            # Combine patch + positional encoding
                            patch_with_pos = np.concatenate([patch.flatten(), pos_enc])
                            current_sequence.append(patch_with_pos)
                    
                    # Pad sequence to expected length if needed
                    while len(current_sequence) < expected_sequence_length:
                        current_sequence.append(np.zeros(patch_features))
                    
                    batch_sequences.append(current_sequence)
                    pbar.update(1)
                
                # Add batch to results and clean up memory
                all_sequences.extend(batch_sequences)
                del batch_sequences, batch_features
        
        print(f"✅ Created {len(all_sequences)} patch sequences with shape: ({expected_sequence_length}, {patch_features})")
        return np.array(all_sequences), labels
    
    def _resize_feature(self, feature, target_height, target_width):
        """Helper to resize/pad features to target dimensions."""
        if feature.shape == (target_height, target_width):
            return feature
        
        # Adjust height
        if feature.shape[0] != target_height:
            if feature.shape[0] < target_height:
                feature = np.pad(feature, ((0, target_height - feature.shape[0]), (0, 0)), mode='constant')
            else:
                feature = feature[:target_height, :]
        
        # Adjust width
        if feature.shape[1] != target_width:
            if feature.shape[1] < target_width:
                feature = np.pad(feature, ((0, 0), (0, target_width - feature.shape[1])), mode='constant')
            else:
                feature = feature[:, :target_width]
        
        return feature

    def resample_audio(self, download_dir="downloads", resampled_dir="training_data"):
        """
        For each .wav file in download_dir, resample to 44100 Hz if needed, and copy to resampled_dir
        in subfolders for both English and scientific names.
        Now includes optional noise reduction preprocessing.
        """
        if input("Do you want to resample audio files? (y/n): ").strip().lower() != 'y':
            print("Skipping audio resampling.")
            return
        
        # Ask about noise reduction
        self.ask_noise_reduction_preference()
            
        if not os.path.exists(resampled_dir):
            os.makedirs(resampled_dir)
        else:
            print(f"Directory {resampled_dir} already exists. Deleting entire folder.")
            shutil.rmtree(resampled_dir)
            os.makedirs(resampled_dir)

        print(f"Scanning directory: {download_dir}")
        all_filenames = [f for f in os.listdir(download_dir) if f.endswith(".wav")]
        
        # Apply dev mode limiting if enabled
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
                    # If parsing fails, skip this file in dev mode
                    continue
            
            print(f"Selected {len(filenames)} files from {len(species_file_counts)} species for resampling")
        else:
            filenames = all_filenames
        
        failed_files = 0
        skipped_files = 0
        processed_files = 0

        print(f"Found {len(filenames)} .wav files to process")
        
        # Progress bar for processing files
        with tqdm(total=len(filenames), desc="Processing audio files", unit="file") as pbar:
            for filename in filenames:
                path = os.path.join(download_dir, filename)
                pbar.set_postfix_str(f"Current: {filename[:30]}...")
                
                try:
                    # Parse filename
                    parts = filename[:-4].split('_')
                    if len(parts) < 4:  # Need at least file_id, english_name, genus, species
                        tqdm.write(f"❌ Filename format not recognized: {filename}")
                        failed_files += 1
                        pbar.update(1)
                        continue
                        
                    file_id = parts[0]
                    english_name = parts[1]
                    genus = parts[2]
                    species = parts[3]
                    
                    # Quick validation before attempting to load
                    if os.path.getsize(path) == 0:
                        tqdm.write(f"⚠️  Skipping empty file: {filename}")
                        skipped_files += 1
                        pbar.update(1)
                        continue
                    
                    # Load and validate the audio file with error handling
                    try:
                        y, sr = librosa.load(path, sr=None)
                        if len(y) == 0:
                            tqdm.write(f"⚠️  Skipping empty audio: {filename}")
                            skipped_files += 1
                            pbar.update(1)
                            continue
                            
                        duration = librosa.get_duration(y=y, sr=sr)
                    except Exception as load_error:
                        tqdm.write(f"❌ Failed to load {filename}: {str(load_error)}")
                        failed_files += 1
                        pbar.update(1)
                        continue

                    if duration < 0.5:  # Skip files shorter than 0.5 seconds
                        tqdm.write(f"⚠️  Skipping {filename}: Duration {duration:.2f}s too short")
                        skipped_files += 1
                        pbar.update(1)
                        continue

                    # Resample if needed
                    if sr > 22050:
                        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
                        sr = 44100
                    elif sr < 22050:
                        tqdm.write(f"⚠️  Skipping {filename}: Sample rate {sr} too low")
                        skipped_files += 1
                        pbar.update(1)
                        continue
                    
                    # Apply noise reduction if enabled
                    if self.use_noise_reduction:
                        y = self.apply_noise_reduction(y, sr)
                        
                    # Save to nested folder
                    nested_folder = os.path.join(resampled_dir, english_name)
                    os.makedirs(nested_folder, exist_ok=True)
                    out_path = os.path.join(nested_folder, filename)
                    sf.write(out_path, y, sr)
                    processed_files += 1
                    
                except KeyboardInterrupt:
                    tqdm.write(f"\n🛑 Process interrupted by user at file: {filename}")
                    break
                except Exception as e:
                    tqdm.write(f"❌ Failed to process {filename}: {str(e)}")
                    failed_files += 1
                
                pbar.update(1)
    
        print(f"\n✅ Resampling complete!")
        print(f"   📊 Processed: {processed_files}")
        print(f"   ⚠️  Skipped: {skipped_files}")
        print(f"   ❌ Failed: {failed_files}")
        print(f"   📁 Total: {len(filenames)}")

def main():
    """Main function to process bird sound data and prepare for ViT training."""
    print("🐦 Bird Sound Classification with Vision Transformer")
    print("=" * 55)
    
    # Setup configuration
    dev_mode = input("Enable dev mode? (10 segments per species for testing) (y/n): ").strip().lower() == 'y'
    
    # Initialize processor
    if dev_mode:
        print("🔧 DEV MODE: Loading limited data for quick testing")
        processor = AudioProcessor(dev_mode=True, dev_limit=10, duration=3)
    else:
        processor = AudioProcessor(duration=3)
    
    print()
    
    # Load and process audio data
    print("📁 Loading audio data...")
    audio_clips, labels = processor.load_audio_data(
        data_dir="training_data", save_segments=True, segments_dir="segments"
    )
    print(f"✅ Loaded {len(audio_clips)} audio clips")
    
    # Encode labels
    print("🏷️  Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(set(encoded_labels))
    print(f"✅ Encoded {num_classes} unique species")
    
    # Split data
    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        audio_clips, encoded_labels, test_size=0.2, random_state=42
    )
    labels_train, labels_test = train_test_split(
        labels, test_size=0.2, random_state=42
    )
    print(f"✅ Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Extract features
    print("🎵 Extracting mel-spectrogram features...")
    mel_features = processor.extract_features(X_train)
    
    # Create ViT patches
    print("🧩 Creating image patches for Vision Transformer...")
    grid_patched_features, patch_labels = processor.create_image_patches_with_position(
        mel_features, y_train, patch_size=16, height=224, width=224
    )
    print(f"✅ Created patches: {np.array(grid_patched_features).shape}")
    
    # Optional: Save spectrograms
    if processor.ask_save_spectrograms_preference():
        processor.save_spectrograms(mel_features, labels_train)
    
    # Optional: Display sample spectrograms
    if input("Display sample spectrograms? (y/n): ").strip().lower() == 'y':
        print("📈 Displaying sample mel-spectrograms...")
        plt.figure(figsize=(20, 8))
        for i in range(min(10, len(mel_features))):
            plt.subplot(2, 5, i + 1)
            librosa.display.specshow(mel_features[i], sr=22050, x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Sample {i + 1}")
        plt.tight_layout()
        plt.show()
    
    print("\n🎯 Data preparation complete! Ready for ViT model training.")
    print(f"   📊 Patch shape: {np.array(grid_patched_features).shape}")
    print(f"   🐦 Classes: {num_classes}")
    print(f"   📈 Training samples: {len(X_train)}")
    
    if dev_mode:
        print(f"\n🔧 DEV MODE SUMMARY:")
        print(f"   • Limited to {processor.dev_limit} files per species during resampling")
        print(f"   • Limited to {processor.dev_limit} files per species during segmentation")
        print(f"   • Limited to {processor.dev_limit} spectrograms per species when saving")
        print(f"   • Used smaller batch sizes for faster processing")
        if hasattr(processor, 'use_noise_reduction') and processor.use_noise_reduction:
            print(f"   • Noise reduction was applied during preprocessing")
        print(f"   This ensures faster testing with representative data from each species.")

if __name__ == "__main__":
    main()