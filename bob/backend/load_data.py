import os
import json
from datasets import load_from_disk

VALID_INDICES_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "valid_indices.json")

def load_dataset(app):
    # Load dataset only once
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        dataset_path = os.path.join(app.root_path, "..", "data", "mywikiart")
        ds = load_from_disk(dataset_path)
        print("✅ Local dataset loaded from disk.")

        # Precompute valid indices and save them only if the JSON doesn't exist
        if not os.path.exists(VALID_INDICES_FILE):
            print("⚙️ Computing valid indices (excluding genre 5)...")
            valid_indices = [
                i for i, sample in enumerate(ds)
                if sample.get("genre") != 5 # Exclude nude paintings
            ]
            with open(VALID_INDICES_FILE, "w") as f:
                json.dump(valid_indices, f)
            print(f"✅ Saved {len(valid_indices)} valid indices to {VALID_INDICES_FILE}")
        else:
            print(f"📄 Found existing valid_indices.json")

        # Load the indices regardless
        with open(VALID_INDICES_FILE, "r") as f:
            app.valid_indices = json.load(f)

        return ds

    else:
        return None
