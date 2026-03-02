import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
import torch
from audio_utils import AudioProcessor

def load_vocalsound_from_disk(data_dir, split):
    split_map = {'train':'tr', 'val':'val', 'validation':'val', 'test':'te'}
    split = split_map.get(split, split)

    data_dir = Path(data_dir)
    json_path = data_dir / "datafiles" / f"{split}.json"

    with open(json_path) as f:
        raw_data = json.load(f)

    data = raw_data["data"]  # <-- FIX: extract the list from the "data" key

    labels = pd.read_csv(data_dir / "class_labels_indices_vs.csv")
    label_map = dict(zip(labels['mid'], labels['index']))

    entries = []
    for item in data:
        wav = Path(item['wav'])  # you already have full path
        label = label_map[item['labels']]
        entries.append({"audio": wav, "label": label})

    return entries

class VocalSoundDataset(Dataset):
    def __init__(self, split, config, data_dir):
        self.config = config
        self.processor = AudioProcessor(config)
        self.entries = load_vocalsound_from_disk(data_dir, split)
        self.label_names = ['laughter','sigh','cough','throat_clearing','sneeze','sniff']

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        waveform, sr = torchaudio.load(item['audio'])
        waveform = waveform.mean(0)
        if sr != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.config.sample_rate
            )
        mel = self.processor.process_audio(waveform)
        label = item['label']
        return {
            "mel": mel,
            "label": label,
            "label_name": self.label_names[label]
        }
