"""DPO 训练所需的数据读取与数据集定义。

本模块负责 MFCC Excel、偏好标签 Excel、样本对齐以及训练/验证划分，
让训练入口只关心优化流程本身。
"""

import logging
import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_SEED = 1314


def parse_label_excel(label_path):
    """将单个 chosen 或 rejected 标签 Excel 解析为 10 步类别张量。"""

    label_dataframe = pd.read_excel(label_path)
    label_values = label_dataframe.values[:, 1:].astype(float)[:10, :].ravel()
    if label_values.size != 10:
        raise ValueError(f"Expected 10 labels in {label_path}, but got {label_values.size}.")
    label_tensor = torch.tensor(label_values - 1, dtype=torch.long)
    if torch.any(label_tensor < 0) or torch.any(label_tensor > 4):
        raise ValueError(f"Label values in {label_path} must map to the range [0, 4].")
    return label_tensor


def parse_mfcc_excel(mfcc_path):
    """将单个 MFCC Excel 文件解析为模型输入张量。"""

    mfcc_data = pd.read_excel(mfcc_path, header=None, engine="openpyxl").values.astype(float)
    return torch.tensor(mfcc_data, dtype=torch.float32).unsqueeze(0)


class PreferenceDataset(Dataset):
    """用于读取对齐后的 MFCC、chosen 标签和 rejected 标签的数据集。"""

    def __init__(self, data_dir, split="train", split_ratio=0.8, seed=DEFAULT_SEED, transforms=None):
        self.data_dir = data_dir
        self.mfcc_dir = os.path.join(data_dir, "MFCC_Output")
        self.chosen_dir = os.path.join(data_dir, "Chosen")
        self.rejected_dir = os.path.join(data_dir, "Rejected")
        self.transforms = transforms
        self.split = split

        self._validate_directories()

        mfcc_dict = self._build_file_dict(self.mfcc_dir, "_MFCC.xlsx")
        chosen_dict = self._build_file_dict(self.chosen_dir, ".xlsx")
        rejected_dict = self._build_file_dict(self.rejected_dir, ".xlsx")

        common_ids = sorted(set(mfcc_dict) & set(chosen_dict) & set(rejected_dict))
        if not common_ids:
            raise ValueError(f"No aligned preference samples found under {data_dir}.")

        logging.info("Found %d aligned preference samples.", len(common_ids))
        logging.info(
            "Missing sample counts | MFCC: %d | Chosen: %d | Rejected: %d",
            len((set(chosen_dict) | set(rejected_dict)) - set(mfcc_dict)),
            len((set(mfcc_dict) | set(rejected_dict)) - set(chosen_dict)),
            len((set(mfcc_dict) | set(chosen_dict)) - set(rejected_dict)),
        )

        shuffled_ids = list(common_ids)
        random.Random(seed).shuffle(shuffled_ids)

        split_index = int(len(shuffled_ids) * split_ratio)
        if split_index <= 0 or split_index >= len(shuffled_ids):
            raise ValueError(
                f"Dataset split would create an empty train or validation set. Total samples: {len(shuffled_ids)}."
            )

        if split == "train":
            selected_ids = shuffled_ids[:split_index]
        elif split == "val":
            selected_ids = shuffled_ids[split_index:]
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.samples = [
            {
                "sample_id": sample_id,
                "mfcc_path": os.path.join(self.mfcc_dir, mfcc_dict[sample_id]),
                "chosen_path": os.path.join(self.chosen_dir, chosen_dict[sample_id]),
                "rejected_path": os.path.join(self.rejected_dir, rejected_dict[sample_id]),
            }
            for sample_id in selected_ids
        ]

        if not self.samples:
            raise ValueError(f"No samples available in split '{split}'.")

        logging.info("Prepared %d %s samples.", len(self.samples), split)

    def _validate_directories(self):
        for directory in [self.mfcc_dir, self.chosen_dir, self.rejected_dir]:
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Required directory does not exist: {directory}")

    @staticmethod
    def _build_file_dict(directory, suffix):
        file_dict = {}
        for filename in os.listdir(directory):
            if not filename.endswith(suffix):
                continue
            sample_id = filename[: -len(suffix)]
            file_dict[sample_id] = filename
        return file_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        mfcc_tensor = parse_mfcc_excel(sample["mfcc_path"])
        chosen_label = parse_label_excel(sample["chosen_path"])
        rejected_label = parse_label_excel(sample["rejected_path"])

        if self.transforms is not None:
            mfcc_tensor = self.transforms(mfcc_tensor)

        return mfcc_tensor, chosen_label, rejected_label, sample["sample_id"]
