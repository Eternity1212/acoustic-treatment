"""Batch score MFCC inputs and write per-sample label Excel files.

This script runs inference over MFCC_Output and writes one label file per
sample. The label file keeps the same headers and dimension names as a
teacher label template, while replacing the score column with model predictions.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from _2_CAM_S import CAMPPlus


#DATASET_ROOT = "/home/zx/Valentin_workplace/网站数据下载/3_28_Sopran"
#MFCC_DIR = "/home/zx/Valentin_workplace/网站数据下载/3_28_Sopran/MFCC_Output"
#LABEL_TEMPLATE_DIR = "/home/zx/Valentin_workplace/网站数据下载/3_28_Sopran/Label"
#OUTPUT_LABEL_DIR = "/home/zx/Valentin_workplace/DPO_data/3_28_Sopran/Rejected"
#PRETRAINED_WEIGHTS = "/home/zx/Valentin_workplace/最佳模型成人/5e-5+16+1e-3/Sopran/best_model.pth"


DATASET_ROOT = "/home/zx/Valentin_workplace/DPO_data/Mezzo"
MFCC_DIR = "/home/zx/Valentin_workplace/DPO_data/Mezzo/train/MFCC_Output/"
LABEL_TEMPLATE_DIR = "/home/zx/Valentin_workplace/DPO_data/Mezzo/train/Chosen/"
OUTPUT_LABEL_DIR = "/home/zx//Valentin_workplace/DPO_data/Mezzo/train/Rejected/"
PRETRAINED_WEIGHTS = "/home/zx/codexProject/vocal_analysis/sft/best_models/Mezzo/2026-03-31_11-21-48/best_model.pth"



VAL_BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_CLASSES = 50

MODEL_TECH_NAMES = [
    "Vibrato",
    "Throat",
    "Position",
    "Open",
    "Clean",
    "Resonate",
    "Unify",
    "Falsetto",
    "Chest",
    "Nasal",
]

LABEL_ALIAS = {
    "Passagio": "Unify",
    "Passaggio": "Unify",
    "Unify": "Unify",
}


class CustomDataset(Dataset):
    """Dataset that reads MFCC Excel files for inference."""

    def __init__(self, mfcc_dir: str):
        self.mfcc_dir = mfcc_dir
        self.mfcc_files = sorted([f for f in os.listdir(self.mfcc_dir) if f.endswith("_MFCC.xlsx")])
        if not self.mfcc_files:
            raise ValueError(f"No MFCC files found in {self.mfcc_dir}.")
        print(f"Found {len(self.mfcc_files)} MFCC files for inference")

    def __len__(self) -> int:
        return len(self.mfcc_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        mfcc_file = self.mfcc_files[idx]
        sample_id = mfcc_file.replace("_MFCC.xlsx", "")
        file_path = os.path.join(self.mfcc_dir, mfcc_file)

        mfcc_data = pd.read_excel(file_path, header=None, engine="openpyxl").values.astype(float)
        mfcc_tensor = torch.tensor(mfcc_data, dtype=torch.float32).unsqueeze(0)
        return mfcc_tensor, sample_id


def is_label_file(filename: str) -> bool:
    if not filename.lower().endswith(".xlsx"):
        return False
    if filename.startswith("~") or filename.startswith(".~") or filename.startswith(".$"):
        return False
    return True


def load_label_template(label_dir: str) -> pd.DataFrame:
    """Load a teacher label file as template (first 10 rows, first 2 columns)."""

    for filename in sorted(os.listdir(label_dir)):
        if not is_label_file(filename):
            continue
        template_path = os.path.join(label_dir, filename)
        df = pd.read_excel(template_path)
        if df.shape[1] < 2:
            continue
        df = df.iloc[:10, :2].copy()
        if len(df) != 10:
            continue
        return df
    raise FileNotFoundError(f"No valid label template found in {label_dir}.")


def build_label_indices(template_df: pd.DataFrame) -> List[int]:
    """Map template label order to model output indices."""

    label_raws = template_df.iloc[:10, 0].tolist()
    indices: List[int] = []
    for raw_label in label_raws:
        label_key = str(raw_label).strip()
        label_key = LABEL_ALIAS.get(label_key, label_key)
        if label_key not in MODEL_TECH_NAMES:
            raise ValueError(f"Unexpected label name '{raw_label}' not in model tech list.")
        indices.append(MODEL_TECH_NAMES.index(label_key))
    return indices


def save_predictions_to_excel(
    net: torch.nn.Module, val_loader: DataLoader, device: torch.device, output_path: str
) -> Tuple[List[str], np.ndarray]:
    """Run inference and save summary predictions to results.xlsx."""

    net.eval()
    all_preds: List[np.ndarray] = []
    all_filenames: List[str] = []

    with torch.no_grad():
        for im, sample_ids in val_loader:
            im = im.to(device)
            output, _, _ = net(im)
            output = output.view(output.shape[0], 5, 10)
            preds = output.argmax(dim=1).cpu().numpy() + 1
            all_preds.append(preds)
            all_filenames.extend(sample_ids)

    if not all_preds:
        raise RuntimeError("No predictions were generated.")

    all_preds = np.concatenate(all_preds, axis=0)
    df_filename = pd.DataFrame({"Filename": all_filenames})
    df_pred = pd.DataFrame(all_preds, columns=[f"Pred_{t}" for t in MODEL_TECH_NAMES])
    df_final = pd.concat([df_filename, df_pred], axis=1).sort_values(by="Filename")
    df_final.to_excel(output_path, index=False)
    print(f"Saved summary predictions to: {output_path}")
    return df_final["Filename"].tolist(), all_preds


def write_label_files(
    template_df: pd.DataFrame, label_indices: List[int], sample_ids: List[str], preds: np.ndarray, output_dir: str
) -> None:
    """Write one label Excel per sample using the template headers and order."""

    os.makedirs(output_dir, exist_ok=True)
    for sample_id, pred_row in zip(sample_ids, preds, strict=True):
        ordered_scores = [float(pred_row[idx]) for idx in label_indices]
        out_df = template_df.copy()
        out_df.iloc[:, 1] = pd.Series(ordered_scores, dtype="float64")
        out_path = os.path.join(output_dir, f"{sample_id}.xlsx")
        out_df.to_excel(out_path, index=False)


def main() -> None:
    if not os.path.isdir(MFCC_DIR):
        raise FileNotFoundError(f"MFCC directory does not exist: {MFCC_DIR}")
    if not os.path.isdir(LABEL_TEMPLATE_DIR):
        raise FileNotFoundError(f"Label template directory does not exist: {LABEL_TEMPLATE_DIR}")

    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    dataset = CustomDataset(MFCC_DIR)
    loader = DataLoader(dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(
        num_class=NUM_CLASSES,
        input_size=1,
        embd_dim=2560,
        growth_rate=64,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
    ).to(device)

    if os.path.exists(PRETRAINED_WEIGHTS):
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device))
        print(f"Loaded pretrained weights from {PRETRAINED_WEIGHTS}")
    else:
        raise FileNotFoundError(f"Pretrained weights not found: {PRETRAINED_WEIGHTS}")

    template_df = load_label_template(LABEL_TEMPLATE_DIR)
    label_indices = build_label_indices(template_df)

    results_path = os.path.join(OUTPUT_LABEL_DIR, "results.xlsx")
    sample_ids, preds = save_predictions_to_excel(model, loader, device, results_path)
    write_label_files(template_df, label_indices, sample_ids, preds, OUTPUT_LABEL_DIR)

    print(f"Wrote {len(sample_ids)} label files to: {OUTPUT_LABEL_DIR}")


if __name__ == "__main__":
    main()
