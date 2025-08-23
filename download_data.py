'''
This script downloads bird recordings from the xeno-canto API, filters by bird types, saves metadata, and generates summaries.
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
        "max_workers": 4
    }

    def __init__(self, country=None, group=None, quality=None, output_dir=None, bird_types=None):
        self.country = country or self.CONFIG["default_country"]
        self.group = group or self.CONFIG["default_group"]
        self.quality = quality or self.CONFIG["default_quality"]
        self.output_dir = output_dir or self.CONFIG["output_dir"]
        self.bird_types = bird_types or {}
        self.api_key = os.getenv("XC_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Set XC_API_KEY in .env file.")
        os.makedirs(self.output_dir, exist_ok=True)

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
            subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ar", "22050", "-ac", "1", dest_path],
                          check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"❌ ffmpeg conversion failed: {e}")

    def download_recordings(self, recordings, max_workers=None):
        """Download recordings to output_dir with optional parallelization."""
        max_workers = max_workers or self.CONFIG["max_workers"]
        downloaded = []

        def download_one(rec):
            if "file" not in rec or not rec.get("length", "0:00"):
                return None
            file_id, file_url = rec["id"], rec["file"]
            en = self._sanitize(rec.get("en", "unknown")).replace("north_island_", "").replace("south_island_", "").replace("new_zealand_", "")
            filename = f"{file_id}_{en}_{self._sanitize(rec.get('gen', 'unknown'))}_{self._sanitize(rec.get('sp', 'unknown'))}_{self._sanitize(rec.get('type', 'call'))}.wav"
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
                        print(f"✅ Saved: {filename}")
                        return filepath
                except Exception as e:
                    print(f"❌ Error downloading {file_id} (attempt {attempt + 1}): {e}")
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm.tqdm(executor.map(download_one, recordings), total=len(recordings), desc="Downloading recordings"))
        downloaded = [r for r in results if r]
        print(f"📥 Downloaded {len(downloaded)} recordings to {self.output_dir}")
        return downloaded

    def _prepare_data(self, recordings):
        """Prepare DataFrame for metadata and summary."""
        data = [
            {
                "id": rec["id"],
                "species": f"{rec.get('gen', 'Unknown')} {rec.get('sp', 'Unknown')}",
                "english_name": rec.get("en", "Unknown").replace("North Island ", "").replace("South Island ", "").replace("New Zealand ", ""),
                "generic_name": rec.get("gen", "Unknown"),
                "scientific_name": rec.get("sp", "Unknown"),
                "sex": rec.get("sex", "Unknown"),
                "file_url": rec["file"],
                "length": self._parse_length(rec.get("length", "0:00"))
            }
            for rec in recordings if "file" in rec and "id" in rec
        ]
        return pd.DataFrame(data)

    def save_metadata(self, recordings, csv_filename="recordings_data.csv"):
        """Save metadata to CSV in logs/."""
        df = self._prepare_data(recordings)
        os.makedirs("logs", exist_ok=True)
        df[["id", "generic_name", "scientific_name", "english_name", "sex", "file_url", "length"]].to_csv(os.path.join("logs", csv_filename), index=False)
        print(f"📊 CSV saved to logs/{csv_filename}")

    def report_summary(self, recordings):
        """Generate summary of recordings."""
        df = self._prepare_data(recordings)
        if df.empty:
            print("⚠️ No recordings to summarize.")
            return
        print("\n📋 --- Recordings Summary ---")
        print(f"Total recordings: {len(df)}")
        print(f"Unique species: {df['species'].nunique()}")
        print("\nTop 5 species by number of recordings:")
        print(df['english_name'].value_counts().head(5))
        print(f"\nAverage recording length: {df['length'].mean():.2f} seconds")
        if "Identity unknown" in df['english_name'].values:
            print(f"Number of 'Identity unknown' recordings: {df[df['english_name'] == 'Identity unknown'].shape[0]}")

    def filter_recordings(self, recordings):
        """Filter recordings by bird types."""
        if not self.bird_types:
            return recordings
        filtered = []
        for rec in recordings:
            if "gen" not in rec or "sp" not in rec:
                continue
            genus_species = f"{rec['gen']} {rec['sp']}"
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
        recordings = self.fetch_data()
        print(f"Total recordings fetched: {len(recordings)}")
        self.save_metadata(recordings)
        filtered = self.filter_recordings(recordings)
        self.report_summary(filtered)
        if input("Download recordings? (y/n): ").strip().lower() != 'y':
            print("Download skipped.")
            return
        use_parallel = input("Use parallel downloads? (y/n): ").strip().lower() == 'y'
        max_workers = self.CONFIG["max_workers"] if use_parallel else 1
        self.download_recordings(filtered, max_workers)

def main():
    """Main function to run the downloader."""
    print("🐦 Starting Bird Recording Downloader...")
    with open("bird_types.yaml", "r") as f:
        bird_types = yaml.safe_load(f)
    downloader = BirdRecordingDownloader(bird_types=bird_types)
    downloader.run()
    print("✅ Bird Recording Downloader finished.")

if __name__ == "__main__":
    main()