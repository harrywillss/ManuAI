'''
This script downloads bird recordings from the Xeno-Canto API and processes local Kaggle dataset (DOC_001_Tier1, DOC_002_DuncanBayParinga),
filters by bird types, saves metadata, and generates summaries.
Assumes Kaggle dataset is manually downloaded and unzipped to a local directory (e.g., archive/).
Saves recordings to downloads/ with filenames: {file_id}_{english_name}_{genus}_{species}_{rec_type}.wav
'''
import os
import time
import requests
import pandas as pd
import re
import tqdm
from dotenv import load_dotenv
import concurrent.futures
import subprocess
import yaml
from pathlib import Path

os.nice(10)

load_dotenv()

class BirdRecordingDownloader:
    BASE_URL = "https://xeno-canto.org/api/3/recordings"
    CONFIG = {
        "recording_types": ["song", "call", "territorial call", "alarm call"],
        "default_country": "New Zealand",
        "default_group": "birds",
        "default_quality": ">C",
        "output_dir": "downloads",
        "max_workers": 4,
        "kaggle_sources": ["DOC_001_Tier1", "DOC_002_DuncanBayParinga"]
    }

    def __init__(self, country=None, group=None, quality=None, output_dir=None, bird_types=None):
        self.country = country or self.CONFIG["default_country"]
        self.group = group or self.CONFIG["default_group"]
        self.quality = quality or self.CONFIG["default_quality"]
        self.output_dir = output_dir or self.CONFIG["output_dir"]
        self.bird_types = bird_types or {}
        self.scientific_to_english = {}
        if self.bird_types:
            for eng, sciences in self.bird_types.items():
                for sci in sciences:
                    self.scientific_to_english[sci.lower()] = eng.lower()
        self.api_key = os.getenv("XC_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Set XC_API_KEY in .env file.")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def _build_query(self, rec_type):
        return f'cnt:"{self.country}" grp:"{self.group}" type:"{rec_type}" q:"{self.quality}"'

    def _sanitize(self, text):
        text = str(text).lower().replace(" ", "_")
        return re.sub(r'[^a-z0-9_]', '', text)

    def _parse_length(self, length_str):
        try:
            mins, secs = map(int, length_str.split(":"))
            return mins * 60 + secs
        except Exception:
            return 0

    def _get_audio_duration(self, filepath):
        """Get duration of an audio file using ffmpeg."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0

    def fetch_data(self, rec_types=None):
        """Fetch recordings for given recording types, handling pagination."""
        rec_types = rec_types or self.CONFIG["recording_types"]
        all_recordings = []
        for rec_type in rec_types:
            page = 1
            query = self._build_query(rec_type)
            print(f"🔍 Fetching {rec_type} recordings with query: {query}")
            while True:
                response = requests.get(self.BASE_URL, params={"query": query, "key": self.api_key, "page": page})
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch data: {response.status_code}")
                data = response.json()
                if page == 1:
                    print(f"✅ Fetched {data['numRecordings']} {rec_type} recordings, {data['numSpecies']} species")
                all_recordings.extend(data.get("recordings", []))
                if page >= int(data.get("numPages", 1)):
                    break
                page += 1
        return list({rec["id"]: rec for rec in all_recordings if "id" in rec}.values())

    def _convert_to_wav_ffmpeg(self, src_path, dest_path):
        """Convert audio to WAV using ffmpeg."""
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", src_path, "-ar", "22050", "-ac", "1", dest_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"❌ ffmpeg conversion failed for {src_path}: {e}")

    def download_xeno_canto_recordings(self, recordings, max_workers=None):
        """Download Xeno-Canto recordings to output_dir with optional parallelization."""
        max_workers = max_workers or self.CONFIG["max_workers"]
        downloaded = []

        def download_one(rec):
            if "file" not in rec or not rec.get("length", "0:00"):
                return None
            file_id, file_url = rec["id"], rec["file"]
            en = self._sanitize(rec.get("en", "unknown")).replace("north_island_", "").replace("south_island_", "").replace("new_zealand_", "")
            genus_species = f"{rec.get('gen', '')} {rec.get('sp', '')}".strip()
            if genus_species and genus_species.lower() in self.scientific_to_english:
                en = self.scientific_to_english[genus_species.lower()]
            filename = f"{file_id}_{en}_{self._sanitize(rec.get('gen', 'unknown'))}_{self._sanitize(rec.get('sp', 'unknown'))}.wav"
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                return filepath
            for attempt in range(3):
                try:
                    resp = requests.get(file_url)
                    if resp.status_code == 200:
                        ext = os.path.splitext(file_url)[1].lower()
                        temp_path = filepath if ext == ".wav" else filepath.replace(".wav", ext)
                        with open(temp_path, "wb") as f:
                            f.write(resp.content)
                        if ext != ".wav":
                            self._convert_to_wav_ffmpeg(temp_path, filepath)
                            os.remove(temp_path)
                        #print(f"✅ Saved: {filename}")
                        return filepath
                except Exception as e:
                    print(f"❌ Error downloading {file_id} (attempt {attempt + 1}): {e}")
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm.tqdm(executor.map(download_one, recordings), total=len(recordings), desc="Downloading Xeno-Canto recordings"))
        downloaded = [r for r in results if r]
        print(f"📥 Downloaded {len(downloaded)} Xeno-Canto recordings to {self.output_dir}")
        return downloaded

    def process_kaggle_recordings(self, kaggle_path, max_workers=None):
        """Process Kaggle dataset recordings from local directory."""
        max_workers = max_workers or self.CONFIG["max_workers"]
        kaggle_root = Path(kaggle_path)
        naming_maps = []
        source_dirs = []

        # Load bird naming maps and find source directories
        for source in self.CONFIG["kaggle_sources"]:
            source_path = kaggle_root / source / source  # Nested structure: e.g., DOC_001_Tier1/DOC_001_Tier1
            if not source_path.exists():
                print(f"⚠️ Source directory not found: {source_path}")
                continue
            bird_map_path = source_path / "bird_naming_map.csv"
            if not bird_map_path.exists():
                print(f"⚠️ bird_naming_map.csv not found in {source_path}. Skipping.")
                continue
            naming_maps.append(pd.read_csv(bird_map_path))
            source_dirs.append(source_path)

        if not naming_maps:
            print("❌ No bird_naming_map.csv found in any Kaggle source directory")
            return [], pd.DataFrame()

        # Combine naming maps, dropping duplicates based on eBird code
        naming_map = pd.concat(naming_maps, ignore_index=True).drop_duplicates(subset=['eBird'])
        print(f"📊 Loaded naming map with {len(naming_map)} entries")

        # Create mapping from eBird to (scientific name, common name)
        ebird_to_info = {
            row['eBird']: (row['ScientificName'], row['CommonName'])
            for _, row in naming_map.iterrows()
        }

        # Collect all FLAC files from train_audio subdirectories
        all_flac_files = []
        for source_dir in source_dirs:
            train_audio_path = source_dir / "train_audio"
            if not train_audio_path.exists():
                print(f"⚠️ train_audio not found in {source_dir}. Skipping.")
                continue
            for species_dir in train_audio_path.iterdir():
                if species_dir.is_dir():
                    ebird_name = species_dir.name
                    for flac_file in species_dir.glob("*.flac"):
                        all_flac_files.append((flac_file, ebird_name))

        print(f"🔍 Found {len(all_flac_files)} FLAC files in Kaggle dataset")

        # Filter based on bird types if specified
        if self.bird_types:
            filtered_flac_files = []
            for flac_path, ebird_name in all_flac_files:
                if ebird_name in ebird_to_info:
                    scientific_name, _ = ebird_to_info[ebird_name]
                    for _, scientific_names in self.bird_types.items():
                        if isinstance(scientific_names, list):
                            if scientific_name in scientific_names:
                                filtered_flac_files.append((flac_path, ebird_name))
                                break
                        elif scientific_name == scientific_names:
                            filtered_flac_files.append((flac_path, ebird_name))
                            break
            all_flac_files = filtered_flac_files
            print(f"✅ Filtered to {len(all_flac_files)} files based on bird types")

        # Assign sequential numeric IDs
        flac_files_with_ids = [(flac_path, ebird_name, str(i + 1)) for i, (flac_path, ebird_name) in enumerate(all_flac_files)]

        # Process files
        def process_one_flac(args):
            flac_path, ebird_name, file_id = args
            if ebird_name not in ebird_to_info:
                return None
            scientific_name, common_name = ebird_to_info[ebird_name]
            if scientific_name.lower() in self.scientific_to_english:
                common_name = self.scientific_to_english[scientific_name.lower()]
            parts = scientific_name.split()
            genus = parts[0] if len(parts) > 0 else "unknown"
            species = parts[1] if len(parts) > 1 else "unknown"
            common_name_clean = self._sanitize(common_name)
            genus_clean = self._sanitize(genus)
            species_clean = self._sanitize(species)
            filename = f"{file_id}_{common_name_clean}_{genus_clean}_{species_clean}.wav"
            filepath = Path(self.output_dir) / filename
            if filepath.exists():
                return str(filepath)
            try:
                self._convert_to_wav_ffmpeg(str(flac_path), str(filepath))
                #print(f"✅ Saved: {filename}")
                return str(filepath)
            except Exception as e:
                print(f"❌ Error converting {flac_path}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm.tqdm(
                executor.map(process_one_flac, flac_files_with_ids),
                total=len(all_flac_files),
                desc="Processing Kaggle recordings"
            ))
        downloaded = [r for r in results if r]
        print(f"📥 Processed {len(downloaded)} Kaggle recordings to {self.output_dir}")
        return downloaded, naming_map

    def _prepare_xeno_canto_data(self, recordings):
        """Prepare DataFrame for Xeno-Canto metadata."""
        data = [
            {
                "id": rec["id"],
                "species": f"{rec.get('gen', 'Unknown')} {rec.get('sp', 'Unknown')}",
                "english_name": rec.get("en", "Unknown").replace("North Island ", "").replace("South Island ", "").replace("New Zealand ", ""),
                "generic_name": rec.get("gen", "Unknown"),
                "scientific_name": rec.get("sp", "Unknown"),
                "sex": rec.get("sex", "Unknown"),
                "file_url": rec["file"],
                "length": self._parse_length(rec.get("length", "0:00")),
                "license": rec.get("lic", "unknown").replace(" ", "_"),
                "location": rec.get("loc", "unknown").replace(" ", "_"),
                "also": rec.get("also", []),
                "smp": rec.get("smp", "unknown"),
                "seen": rec.get("animal-seen", "unknown"),
                "regnr": rec.get("regnr", "unknown"),
                "file": rec.get("file", "unknown"),
                "source": "Xeno-Canto"
            }
            for rec in recordings if "file" in rec and "id" in rec
        ]
        return pd.DataFrame(data)

    def _prepare_kaggle_data(self, kaggle_path, naming_map):
        """Prepare DataFrame for Kaggle metadata."""
        kaggle_root = Path(kaggle_path)
        data = []
        file_id_counter = 1
        for source in self.CONFIG["kaggle_sources"]:
            source_path = kaggle_root / source / source
            if not source_path.exists():
                continue
            train_audio_path = source_path / "train_audio"
            if not train_audio_path.exists():
                continue
            for species_dir in train_audio_path.iterdir():
                if species_dir.is_dir():
                    ebird_name = species_dir.name
                    naming_row = naming_map[naming_map['eBird'] == ebird_name]
                    if naming_row.empty:
                        continue
                    common_name = naming_row.iloc[0]['CommonName']
                    scientific_name_full = naming_row.iloc[0]['ScientificName']
                    if scientific_name_full.lower() in self.scientific_to_english:
                        common_name = self.scientific_to_english[scientific_name_full.lower()]
                    scientific_name = naming_row.iloc[0]['ScientificName'].split()
                    genus = scientific_name[0] if len(scientific_name) > 0 else "unknown"
                    species = scientific_name[1] if len(scientific_name) > 1 else "unknown"
                    for flac_file in species_dir.glob("*.flac"):
                        file_id = str(file_id_counter)
                        file_id_counter += 1
                        filename = f"{file_id}_{self._sanitize(common_name)}_{self._sanitize(genus)}_{self._sanitize(species)}.wav"
                        filepath = Path(self.output_dir) / filename
                        data.append({
                            "id": file_id,
                            "species": f"{genus} {species}",
                            "english_name": common_name,
                            "generic_name": genus,
                            "scientific_name": species,
                            "sex": "unknown",
                            "file_url": str(flac_file),
                            "length": self._get_audio_duration(str(flac_file)),
                            "license": "unknown",
                            "location": source,
                            "also": [],
                            "smp": "32000",
                            "seen": "unknown",
                            "regnr": "unknown",
                            "file": str(filepath),
                            "source": "Kaggle"
                        })
        return pd.DataFrame(data)

    def save_metadata(self, xeno_recordings, kaggle_path, naming_map, csv_filename="recordings_data.csv"):
        """Save combined metadata to CSV in logs/."""
        xeno_df = self._prepare_xeno_canto_data(xeno_recordings)
        kaggle_df = self._prepare_kaggle_data(kaggle_path, naming_map)
        df = pd.concat([xeno_df, kaggle_df], ignore_index=True)
        if df.empty:
            print("⚠️ No metadata to save (empty dataset).")
            return
        os.makedirs("logs", exist_ok=True)
        df[["id", "generic_name", "scientific_name", "english_name", "sex", "file_url", "length", "license", "location", "bird_type", "also", "smp", "seen", "regnr", "file", "source"]].to_csv(
            os.path.join("logs", csv_filename), index=False
        )
        print(f"📊 CSV saved to logs/{csv_filename}")

    def report_summary(self, xeno_recordings, kaggle_path, naming_map):
        """Generate summary of all recordings."""
        xeno_df = self._prepare_xeno_canto_data(xeno_recordings)
        kaggle_df = self._prepare_kaggle_data(kaggle_path, naming_map)
        df = pd.concat([xeno_df, kaggle_df], ignore_index=True)
        if df.empty:
            print("⚠️ No recordings to summarize.")
            return
        print("\n📋 --- Recordings Summary ---")
        print(f"Total recordings: {len(df)}")
        print(f"Xeno-Canto recordings: {len(xeno_df)}")
        print(f"Kaggle recordings: {len(kaggle_df)}")
        print(f"Unique species: {df['species'].nunique()}")
        print("\nTop 5 species by number of recordings:")
        print(df['english_name'].value_counts().head(5))
        print(f"\nAverage recording length: {df['length'].mean():.2f} seconds")
        if "Identity unknown" in df['english_name'].values:
            print(f"Number of 'Identity unknown' recordings: {df[df['english_name'] == 'Identity unknown'].shape[0]}")

    def filter_recordings(self, recordings, source="Xeno-Canto"):
        """Filter recordings by bird types."""
        if not self.bird_types:
            return recordings
        filtered = []
        for rec in recordings:
            if source == "Xeno-Canto":
                if "gen" not in rec or "sp" not in rec:
                    continue
                genus_species = f"{rec['gen']} {rec['sp']}"
            else:  # Kaggle
                genus_species = rec['species']
            for _, scientific_names in self.bird_types.items():
                if isinstance(scientific_names, list):
                    if genus_species in scientific_names:
                        filtered.append(rec)
                        break
                elif genus_species == scientific_names:
                    filtered.append(rec)
                    break
        return filtered

    def run(self):
        """Run the downloader pipeline."""
        print("🐦 Starting Bird Recording Downloader...")

        xeno_recordings = []
        kaggle_path = ""
        naming_map = pd.DataFrame()
        max_workers = self.CONFIG["max_workers"]

        # Option to load Xeno-Canto recordings
        if input("Load Xeno-Canto recordings? (y/n): ").strip().lower() == 'y':
            xeno_recordings = self.fetch_data()
            print(f"Total Xeno-Canto recordings fetched: {len(xeno_recordings)}")
            xeno_filtered = self.filter_recordings(xeno_recordings, source="Xeno-Canto")
            self.download_xeno_canto_recordings(xeno_filtered, max_workers)
        else:
            xeno_filtered = []

        # Option to process Kaggle dataset recordings
        if input("Process Kaggle dataset recordings? (y/n): ").strip().lower() == 'y':
            kaggle_path = "archive"  # Default path; change if needed
            if not os.path.exists(kaggle_path):
                print(f"❌ Kaggle dataset directory does not exist: {kaggle_path}")
            else:
                use_parallel = input("Use parallel processing for Kaggle dataset? (y/n): ").strip().lower() == 'y'
                max_workers = self.CONFIG["max_workers"] if use_parallel else 1
                downloaded, naming_map = self.process_kaggle_recordings(kaggle_path, max_workers)
        else:
            kaggle_filtered = []

        # Save combined metadata and report summary
        self.save_metadata(xeno_filtered, kaggle_path, naming_map)
        self.report_summary(xeno_filtered, kaggle_path, naming_map)
        print("✅ Bird Recording Downloader finished.")

def main():
    """Main function to run the downloader."""
    with open("bird_types.yaml", "r") as f:
        bird_types = yaml.safe_load(f)
    downloader = BirdRecordingDownloader(bird_types=bird_types)
    downloader.run()
    print("✅ Bird Recording Downloader finished.")

if __name__ == "__main__":
    main()