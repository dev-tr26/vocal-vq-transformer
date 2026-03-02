import json
from pathlib import Path

json_dir = Path(r"C:\Desktop\datasets\audio_16_VocalSound\datafiles")
audio_base = Path(r"C:\Desktop\datasets\audio_16_VocalSound\audio_16k")
json_files = ["tr.json", "val.json", "te.json", "all.json"]

for jf in json_files:
    json_path = json_dir / jf
    with open(json_path, "r") as f:
        data = json.load(f)
    
    for item in data["data"]:
        filename = Path(item["wav"]).name
        item["wav"] = str(audio_base / filename)
    
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {jf} successfully.")
