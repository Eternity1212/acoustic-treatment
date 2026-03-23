# 04_evaluate_alignment.py
import torch
import numpy as np
import pandas as pd
import os
from CAM_S import CAMPPlus
from pathlib import Path

# ============ 路径设置 ============
data_dir = r"D:\比赛视频Videos\sopran_cutted"
mfcc_dir = os.path.join(data_dir, "MFCC_Output")
label_dir = os.path.join(data_dir, "Label")
model_path = os.path.join(data_dir, "logs_dpo", "best_dpo_model.pth")
output_dir = os.path.join(data_dir, "results")
os.makedirs(output_dir, exist_ok=True)

# ============ 加载模型 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAMPPlus(
    num_class=10,
    input_size=1,
    embd_dim=8192,
    growth_rate=64,
    bn_size=4,
    init_channels=128,
    config_str='batchnorm-relu'
).to(device)

model.output_1 = nn.Linear(8192, 10).to(device)  # 确保是回归层
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ============ 加载测试数据 ============
test_files = []
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".xlsx"):
        continue
    test_files.append(label_file.replace(".xlsx", ""))

predictions = []
ground_truths = []

for file in test_files:
    mfcc_path = os.path.join(mfcc_dir, file + "_MFCC.pt")
    label_path = os.path.join(label_dir, file + ".xlsx")
    if not os.path.exists(mfcc_path) or not os.path.exists(label_path):
        continue

    mfcc = torch.load(mfcc_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output, _, _ = model(mfcc)
        pred = output.cpu().numpy()[0]  # (10,)

    df = pd.read_excel(label_path, header=None, engine="openpyxl")
    true_scores = df.iloc[0, :10].values  # 前10维

    predictions.append(pred)
    ground_truths.append(true_scores)

# ============ 评估指标 ============
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

# 1. MSE
mse = np.mean((predictions - ground_truths) ** 2)
print(f"✅ MSE: {mse:.4f}")

# 2. Spearman 相关系数（每个维度）
from scipy.stats import spearmanr

spearman_scores = []
for i in range(10):
    rho, p = spearmanr(predictions[:, i], ground_truths[:, i])
    spearman_scores.append(rho)
    print(f"维度 {i+1} Spearman ρ: {rho:.4f}")

avg_spearman = np.mean(spearman_scores)
print(f"✅ 平均 Spearman ρ: {avg_spearman:.4f}")

# 3. 保存结果
results_df = pd.DataFrame({
    "File": test_files,
    "Pred_Vibrato": predictions[:, 0],
    "Pred_Throat": predictions[:, 1],
    "Pred_Position": predictions[:, 2],
    "Pred_Open": predictions[:, 3],
    "Pred_Clean": predictions[:, 4],
    "Pred_Resonate": predictions[:, 5],
    "Pred_Unify": predictions[:, 6],
    "Pred_Falsetto": predictions[:, 7],
    "Pred_Chset": predictions[:, 8],
    "Pred_Nasal": predictions[:, 9],
    "True_Vibrato": ground_truths[:, 0],
    "True_Throat": ground_truths[:, 1],
    "True_Position": ground_truths[:, 2],
    "True_Open": ground_truths[:, 3],
    "True_Clean": ground_truths[:, 4],
    "True_Resonate": ground_truths[:, 5],
    "True_Unify": ground_truths[:, 6],
    "True_Falsetto": ground_truths[:, 7],
    "True_Chset": ground_truths[:, 8],
    "True_Nasal": ground_truths[:, 9]
})

results_df.to_excel(os.path.join(output_dir, "alignment_results.xlsx"), index=False)
print(f"✅ 评估结果已保存至: {output_dir}")