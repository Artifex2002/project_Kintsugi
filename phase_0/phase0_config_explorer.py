"""
Model Metadata & Action Space Explorer
========================================================
Dynamically interrogates Hugging Face models for their
config.json
While processor files handle the exact mathematical stats
the config.json file serves as the blueprint for the 
model's neural network architecture and data pipeline.
"""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_TO_EXPLORE = [
    "lerobot/smolvla_base",
    "HuggingFaceVLA/smolvla_libero"
]

# Directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Create phase_0 directory in the same location as the script
OUTPUT_DIR = SCRIPT_DIR / "phase0_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def print_raw_config(model_id):
    print(f"\n{'='*70}")
    print(f" RAW CONFIG FOR: {model_id}")
    print(f"{'='*70}")
    
    try:
        # hf_hub_download reaches out to the Hugging Face servers.
        # repo_id: the name of the model on the Hub.
        # filename: the specific metadata file we want to read.
        # repo_type="model": ensures it looks in the Models database, not Datasets.
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            repo_type="model"
        )
        
        # Open the downloaded file in read-only mode ("r")
        with open(config_path, "r") as f:
            # Parse the raw JSON text into a navigable Python dictionary
            config = json.load(f)
            
        # json.dumps converts the dictionary back to a string.
        # indent=2 adds spaces and line breaks so it is easily readable by humans,
        # rather than printing as one massive, unreadable block of text.
        pretty_json = json.dumps(config, indent=2)
        print(pretty_json)

        # Create safe filename
        safe_name = model_id.replace("/", "__")
        output_file = OUTPUT_DIR / f"{safe_name}_config.json"

        # Save file
        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n Saved to: {output_file}")

    except Exception as e:
        print(f"✗ Failed to download or read config.json: {e}")
        
    except Exception as e:
        # If the file doesn't exist, the repo is private, or the internet drops,
        # we catch the error here so the script doesn't completely crash.
        print(f"✗ Failed to download or read config.json: {e}")

# Loop through our list and run the function for each model
for model_id in MODELS_TO_EXPLORE:
    print_raw_config(model_id)