"""Library for rendering META-SiM Atlas."""
import os
import requests
import zipfile
import shutil
import joblib
import warnings


FILENAME = 'atlas_pp_distilled.keras'
FILEPATH = os.path.join(os.path.dirname(__file__), FILENAME)  # Path in the same directory as atlas.py
FILE_URL = "https://storage.googleapis.com/fret_traces/Atlas/atlas_pp.joblib.zip"
DENSITY_MODEL_FILENAME = 'atlas_pp_density_models.joblib'

DENSITY_MODEL_FILEPATH = os.path.join(os.path.dirname(__file__), DENSITY_MODEL_FILENAME)  # Path in the same directory as atlas.py

# Constant for Atlas category names.
SORTED_NAMES = [
 '1-c-l-s',
 '1-n-l-s',
 '1-c-m-s',
 '1-n-m-s',
 '1-c-h-s',
 '1-n-h-s',
 '2-c-lm-s',
 '2-n-lm-s',
 '2-c-mh-s',
 '2-n-mh-s',
 '2-c-lh-s',
 '2-n-lh-s',
 '2-c-lm-f',
 '2-n-lm-f',
 '2-c-mh-f',
 '2-n-mh-f',
 '2-c-lh-f',
 '2-n-lh-f',
 '3-c-lmh-s',
 '3-n-lmh-s',
 '3-c-lmh-f',
 '3-n-lmh-f',
]
# Shorter names for 1-state categories.
CLEANED_NAMES = {
 '1-c-l-s': '1-c-l',
 '1-n-l-s': '1-n-l',
 '1-c-m-s': '1-c-m',
 '1-n-m-s': '1-n-m',
 '1-c-h-s': '1-c-h',
 '1-n-h-s': '1-n-h',
}

BATCH_SIZE = 128


def download_and_unzip_atlas(file_url=FILE_URL, filename=FILENAME):
    """
    Downloads and unzips the Atlas file if it doesn't exist.

    Args:
        file_url: The URL of the atlas.pp.zip file.
        filename: The desired filename for the unzipped file.
    """

    filepath = os.path.join(os.path.dirname(__file__), filename)  # Path in the same directory as atlas.py
    zip_filepath = filepath + ".zip"

    if os.path.exists(filepath):
        return  # No need to download

    if os.path.exists(zip_filepath):
        print(f"Zip file '{filename}.zip' already exists. Skipping Download.")
    else:
        confirm = input(f"Atlas model (3GB) needs to be downloaded. Do you want to download it? (yes/no): ")
        if confirm.lower() != "yes" and confirm.lower() != "y":
            print("Download cancelled.")
            return

        try:
            print(f"Downloading '{filename}' from {file_url}...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            with open(zip_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Download of '{filename}' complete.")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading '{filename}': {e}")
            if os.path.exists(zip_filepath): #remove the partially downloaded zip file
                os.remove(zip_filepath)
            return

    try:
        print(f"Unzipping '{filename}'...")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(filepath))  # Extract to the same directory
        print(f"Unzipping '{filename}' complete.")

        # Optionally, remove the zip file after extraction
        os.remove(zip_filepath)

    except zipfile.BadZipFile as e:
        print(f"Error unzipping '{filename}': {e}")
        if os.path.exists(filepath): #remove the potentially corrupted file
            os.remove(filepath)
    except Exception as e:
        print(f"An unexpected error occurred during unzip: {e}")
        if os.path.exists(filepath): #remove the potentially corrupted file
            os.remove(filepath)


def get_atlas_2d(embedding):
    """Generates 2D Atlas coordinates for traces."""
    import tensorflow.keras as keras
    model = keras.saving.load_model(FILEPATH)
    return model.predict(embedding, batch_size=BATCH_SIZE, verbose=0)


def get_atlas_density_model():
    """Gets the density model for drawing Atlas boundaries."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(DENSITY_MODEL_FILEPATH)

