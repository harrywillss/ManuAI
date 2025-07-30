'''
This script downloads bird recordings from the xeno-canto API based on specified criteria.
It supports filtering by bird types, recording types, and quality, and saves the recordings to a specified directory.
It also generates metadata and summary reports of the recordings.
'''

import os
import requests
import pandas as pd
import re
import tqdm
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables from .env file
load_dotenv()

# Bird types: ["Fantail", "Kākā", "Bellbird", "Tūī", "Kea", "Morepork", "Kākāpō", "House Sparrow"], # List of bird types to search for
# Recording types: ["song", "call"]
# Recording quality: ">C" (C or better)
# Country: "New Zealand"
# Group: "birds"

class BirdRecordingDownloader:
    BASE_URL = "https://xeno-canto.org/api/3/recordings"
    def __init__(self, country="New Zealand", group="birds", recording_type="call", quality=">C", output_dir="downloads", bird_types=None):
        self.country = country
        self.group = group
        self.recording_type = recording_type
        self.quality = quality
        self.output_dir = output_dir
        self.bird_types = bird_types if bird_types else []
        self.api_key = os.getenv("XC_API_KEY")

        if not self.api_key:
            raise ValueError("API key not found. Please set the XC_API_KEY environment variable.")

        os.makedirs(self.output_dir, exist_ok=True)
        self.query_string = self._build_query()

    def _build_query(self):
        # Build the query string for the xeno-canto API
        return f'cnt:"{self.country}" grp:"{self.group}" type:"{self.recording_type}" q:"{self.quality}"'

    def _parse_length(self, length_str):
        # Converts "m:ss" to seconds
        try:
            mins, secs = map(int, length_str.split(":"))
            return mins * 60 + secs
        except Exception:
            return 0

    def fetch_data(self):
        '''
        Fetch data from the xeno-canto API based on the query string.
        Handles pagination and returns all recordings.
        '''
        print("🔍 Fetching data with query:", self.query_string)
        all_recordings = []
        page = 1
        while True:
            response = requests.get(
                self.BASE_URL,
                params={"query": self.query_string, "key": self.api_key, "page": page}
            )
            if response.status_code != 200:
                raise Exception(f"Failed to fetch data: {response.status_code} {response.text}")
            data = response.json()
            if page == 1:
                print(f"✅ Data fetched successfully. Recordings: {data['numRecordings']}, Species: {data['numSpecies']}")
            all_recordings.extend(data.get("recordings", []))
            if page >= int(data.get("numPages", 1)):
                break
            page += 1
        return {"recordings": all_recordings}

    def _sanitize(self, text):
        '''
        Sanitize text for use in filenames.
        Converts to lowercase, replaces spaces with underscores, and removes non-alphanumeric characters.
        '''
        text = str(text).lower().replace(" ", "_")
        return re.sub(r'[^a-z0-9_]', '', text)

    def download_recordings(self, recordings, max_workers=4):
        '''
        Download recordings to the output directory using limited parallel threads.
        '''
        downloaded = []

        def download_one(rec):
            if "file" not in rec or "id" not in rec:
                return None
            file_url = rec["file"]
            file_id = rec["id"]
            gen = self._sanitize(rec.get("gen", "unknown"))
            sp = self._sanitize(rec.get("sp", "unknown"))
            en = self._sanitize(rec.get("en", "unknown"))
            rec_type = self._sanitize(rec.get("type", self.recording_type))
            length = rec.get("length", "0:00")
            if length == "0:00":
                print(f"⚠️ Recording {file_id} has no length data, skipping download.")
                return None

            invalid_names = ["south_island_", "north_island_", "new_zealand_"]
            for name in invalid_names:
                if name in en:
                    en = en.replace(name, "")

            genus_species = f"{gen}_{sp}"
            filename = f"{file_id}_{en}_{genus_species}_{rec_type}.wav"
            filepath = os.path.join(self.output_dir, filename)

            if os.path.exists(filepath):
                print(f"⏩ File already exists: {filename}, skipping download.")
                return filepath

            print(f"⬇️ Downloading {filename}...")
            try:
                resp = requests.get(file_url, timeout=30)
                if resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    print(f"✅ Saved: {filename}")
                    return filepath
                else:
                    print(f"❌ Failed to download {file_id}: {resp.status_code}")
            except Exception as e:
                print(f"❌ Error downloading {file_id}: {e}")
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm.tqdm(executor.map(download_one, recordings), total=len(recordings)))
            downloaded = [r for r in results if r is not None]

        print(f"📥 Downloaded {len(downloaded)} recordings to {self.output_dir}")
        return downloaded

    def save_metadata(self, recordings, csv_filename="recordings_data.csv"):
        '''
        Save metadata of recordings to a CSV file.
        '''
        metadata = {
            "query": self.query_string,
            "total_recordings": len(recordings),
            "recordings": [
                {
                    "id": rec["id"],
                    "generic_name": rec.get("gen", "Unknown"),
                    "scientific_name": rec.get("sp", "Unknown"),
                    "english_name": rec.get("en", "Unknown"),
                    "sex": rec.get("sex", "Unknown"),
                    "file_url": rec["file"],
                    "length": rec.get("length", "Unknown")
                }
                for rec in recordings if "file" in rec and "id" in rec
            ]
        }

        df = pd.DataFrame(metadata["recordings"])
        # Combine North Island and South Island species
        df['english_name'] = df['english_name'].str.replace("North Island ", "", regex=False).str.replace("South Island ", "", regex=False)

        # Remove "New Zealand" from species names
        df['english_name'] = df['english_name'].str.replace("New Zealand ", "", regex=False)
        df.to_csv("logs/" + csv_filename, index=False)
        print(f"📊 CSV data saved to {csv_filename}")

    def report_summary(self, recordings):
        '''
        Generate a summary report of the recordings.
        '''
        if not recordings:
            print("⚠️ No recordings to summarize.")
            return

        df = pd.DataFrame([
            {
                "species": f"{rec.get('gen', 'Unknown')} {rec.get('sp', 'Unknown')}",
                "english_name": rec.get("en", "Unknown"),
                "length": self._parse_length(rec.get("length", "0:00")),
                "sex": rec.get("sex", "Unknown")
            }
            for rec in recordings if "file" in rec and "id" in rec
        ])

        # MAYBE NOT - could use 'Identity unknown' as test data.
        '''# Remove recordings with 'Identity unknown'
        df = df[df['english_name'] != "Identity unknown"]
        if df.empty:
            print("⚠️ No valid recordings found after filtering.")
            return'''
        
        # Combine North Island and South Island species
        df['english_name'] = df['english_name'].str.replace("North Island ", "", regex=False).str.replace("South Island ", "", regex=False)

        # Remove "New Zealand" from species names
        df['english_name'] = df['english_name'].str.replace("New Zealand ", "", regex=False)

        print("\n📋 --- Recordings Summary ---")
        print(f"Total recordings: {len(df)}")
        print(f"Unique species: {df['species'].nunique()}")

        print("\nTop 10 species by number of recordings:")
        print(df['english_name'].value_counts().head(10))

        avg_length = df['length'].mean()
        print(f"\nAverage recording length: {avg_length:.2f} seconds")

        # Number of "Identity unknown" recordings
        identity_unknown_count = df[df['english_name'] == "Identity unknown"].shape[0]
        if identity_unknown_count > 0:
            print(f"\nNumber of 'Identity unknown' recordings: {identity_unknown_count}")

        
    def filter_recordings(self, recordings):
        '''
        Filter recordings based on the specified bird types.
        Only keep recordings that match the bird types defined in the class.
        '''
        filtered = []
        for rec in recordings:
            if "gen" not in rec or "sp" not in rec or "en" not in rec:
                continue
            
            genus_species = f"{rec['gen']} {rec['sp']}"
            english_name = rec.get("en", "Unknown")
            
            # Check if this recording matches any of our target bird types
            found_match = False
            if self.bird_types:
                for bird_name, scientific_names in self.bird_types.items():
                    # Handle both single species and lists of species
                    if isinstance(scientific_names, list):
                        if genus_species in scientific_names:
                            found_match = True
                            break
                    else:
                        if genus_species == scientific_names:
                            found_match = True
                            break
                    """ 
                    # Also check by English name
                    if english_name == bird_name:
                        found_match = True
                        break """
                
                if not found_match:
                    continue
            
            filtered.append(rec)
        return filtered

    def run(self):
        # Fetch 'song' recordings
        self.recording_type = "song"
        self.query_string = self._build_query()
        data_song = self.fetch_data()
        recordings_song = data_song.get("recordings", [])

        # Fetch 'call' recordings
        self.recording_type = "call"
        self.query_string = self._build_query()
        data_call = self.fetch_data()
        recordings_call = data_call.get("recordings", [])

        # Fetch 'territorial call' recordings
        self.recording_type = "territorial call"
        self.query_string = self._build_query()
        data_territorial_call = self.fetch_data()
        recordings_territorial_call = data_territorial_call.get("recordings", [])

        # Fetch 'alarm call' recordings
        self.recording_type = "alarm call"
        self.query_string = self._build_query()
        data_alarm_call = self.fetch_data()
        recordings_alarm_call = data_alarm_call.get("recordings", [])

        # Combine and remove duplicates by 'id'
        all_recordings = {rec['id']: rec for rec in recordings_song + recordings_call +
                          recordings_territorial_call + recordings_alarm_call}.values()
        
        # Report number of recordings fetched and species found
        print(f"Total recordings fetched: {len(all_recordings)}")

        filtered_recordings = self.filter_recordings(all_recordings)

        self.save_metadata(filtered_recordings)
        self.report_summary(filtered_recordings)

        if input("Do you want to download the recordings? (y/n): ").strip().lower() != 'y':
            print("Download skipped.")
            return
        use_parallel = input("Use parallel downloads? (y/n): ").strip().lower() == 'y'
        max_workers = 4 if use_parallel else 1
        print("📥 Downloading filtered recordings...")
        self.download_recordings(filtered_recordings, max_workers=max_workers)

bird_types = {
    "Tūī": "Prosthemadera novaeseelandiae",
    "Bellbird/Korimako": "Anthornis melanura",
    "Fantail/Pīwakawaka": "Rhipidura fuliginosa",
    "Robin/Toutouwai": ["Petroica longipes", "Petroica australis"],
    "Kākā": ["Nestor meridionalis", "Nestor meridionalis septentrionalis", "Nestor meridionalis meridionalis"],
    "Tomtit/Miromiro": "Petroica macrocephala",
    "Whitehead/Pōpokotea": "Mohoua albicilla",
    "Morepork/Ruru": ["Ninox novaeseelandiae", "Ninox boobook"],
    "Saddleback/Tīeke": ["Philesturnus rufusater", "Philesturnus carunculatus"],
    "Silvereye/Tauhou": "Zosterops lateralis"
}

# Use instead of run function in main.py
def main():
    '''
    Main function to run the Bird Recording Downloader.
    Downloads bird recordings based on specified criteria and saves them to the output directory.
    '''
    
    tqdm.tqdm.pandas(desc="Downloading recordings")
    print("🐦 Starting Bird Recording Downloader...")
    downloader = BirdRecordingDownloader(bird_types=bird_types)
    downloader.run()
    print("✅ Bird Recording Downloader finished.")

if __name__ == "__main__":
    main()