from pathlib import Path

import datasets

from uni2ts.common.env import env


class HFDataset:
    def __init__(self, dataset_name: str, storage_path: Path = env.CUSTOM_DATA_PATH):
        self.hf_dataset = datasets.load_from_disk(
            str(storage_path / dataset_name)
        ).with_format("numpy")
        self.freq = self.hf_dataset[0]["freq"]
        self.target_dim = (
            target.shape[-1]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    def __iter__(self):
        for sample in self.hf_dataset:
            sample["start"] = sample["start"].item()
            yield sample


class HFDatasetM:
    def __init__(self, dataset_name: str, storage_path: Path = env.CUSTOM_DATA_PATH):
        self.hf_dataset = datasets.load_from_disk(
            str(storage_path / dataset_name)
        ).with_format("numpy")
        self.freq = self.hf_dataset[0]["freq"]
        self.target_dim = (
            target.shape[0]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

        if 'past_feat_dynamic_real' in self.hf_dataset[0]:
            self.past_feat_dynamic_real_dim = (
                past_feat_dynamic_real.shape[0]
                if len((past_feat_dynamic_real := self.hf_dataset[0]["past_feat_dynamic_real"]).shape) > 1
                else 1
            )
        else:
            self.past_feat_dynamic_real_dim = 0

        if 'feat_dynamic_real' in self.hf_dataset[0]:
            self.feat_dynamic_real_dim = (
                feat_dynamic_real.shape[0]
                if len((feat_dynamic_real := self.hf_dataset[0]["feat_dynamic_real"]).shape) > 1
                else 1
            )
        else:
            self.feat_dynamic_real_dim = 0

    def __iter__(self):
        for sample in self.hf_dataset:
            sample["start"] = sample["start"].item()
            yield sample