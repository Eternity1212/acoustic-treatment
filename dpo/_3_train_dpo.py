# 03_train_dpo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from CAM_S import CAMPPlus  # 你已有的模型
import numpy as np
from tqdm import tqdm

# ============ 超参数 ============
data_dir = r"D:\比赛视频Videos\sopran_cutted"
preference_json = os.path.join(data_dir, "preference_data.json")
log_dir = os.path.join(data_dir, "logs_dpo")
os.makedirs(log_dir, exist_ok=True)

batch_size = 8
num_epochs = 50
learning_rate = 1e-5
temperature = 1.0  # DPO 温度参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============ 加载偏好数据 ============
with open(preference_json, "r", encoding="utf-8") as f:
    preference_data = json.load(f)

print(f"✅ 加载 {len(preference_data)} 个偏好对")

# ============ 自定义 Dataset ============
class PreferenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mfcc_a = torch.tensor(item["mfcc_a"], dtype=torch.float32).unsqueeze(0)  # (1, 128, 128)
        mfcc_b = torch.tensor(item["mfcc_b"], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(item["label"], dtype=torch.float32)  # 1 表示偏好 A
        return mfcc_a, mfcc_b, label

dataset = PreferenceDataset(preference_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ============ 加载模型 ============
model = CAMPPlus(
    num_class=10,  # 改为 10 维回归
    input_size=1,
    embd_dim=8192,
    growth_rate=64,
    bn_size=4,
    init_channels=128,
    config_str='batchnorm-relu'
).to(device)

# 修改输出层为回归（10维）
model.output_1 = nn.Linear(8192, 10).to(device)  # 替换原分类层

# ============ DPO 损失函数 ============
def dpo_loss(model, mfcc_a, mfcc_b, label, temperature=1.0):
    # 前向传播
    output_a, _, _ = model(mfcc_a)  # (B, 10)
    output_b, _, _ = model(mfcc_b)  # (B, 10)

    # 计算 logits（偏好分数）
    logits_a = torch.sum(output_a, dim=1)  # (B,)
    logits_b = torch.sum(output_b, dim=1)  # (B,)

    # DPO 损失
    # 参考：https://arxiv.org/abs/2305.18290
    # loss = -log(sigmoid((logits_a - logits_b) / temperature)) if label==1 else -log(sigmoid((logits_b - logits_a) / temperature))
    # 简化版：直接优化偏好概率
    diff = logits_a - logits_b
    log_prob = torch.log(torch.sigmoid(diff / temperature))
    loss = -log_prob.mean()  # 最大化偏好概率

    return loss

# ============ 优化器 ============
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

# ============ 训练循环 ============
writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for mfcc_a, mfcc_b, label in progress_bar:
        mfcc_a, mfcc_b, label = mfcc_a.to(device), mfcc_b.to(device), label.to(device)

        optimizer.zero_grad()
        loss = dpo_loss(model, mfcc_a, mfcc_b, label, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('DPO_Loss', avg_loss, epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 保存最佳模型
    if avg_loss est_loss:< b
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(log_dir, "best_dpo_model.pth"))
        print(f"✅ 保存最佳模型，Loss: {best_loss:.4f}")

writer.close()
print("🎉 DPO 训练完成！")