import torch
'''
from CAM import CAMPPlus
'''
from _2_CAM_S import CAMPPlus
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
import random

# 定义超参数
'''
data_dir = r"/home/zx/audio/fusai_data/Sopran/zz"  # 数据集路径
'''

data_dir = r"/home/zx/Valentin_workplace/Mezzo_test"  # 数据集路径



val_batch_size = 16  # 验证集批量大小
num_workers = 4  # 多线程数据加载
num_classes = 50  # 分类数量


pretrained_weights = r"/home/zx/AST1/best_models_for_grid_Sopran+Mezzo/1e-5+16+1e-4/2026-03-03_10-05-07/best_model.pth"  # 预训练权重路径

# pretrained_weights = "/home/zx/git_workplace/DIVA-AI-Multimodal-Reinforcement-Learning/OJ_System/models/best_models_for_grid/5e-5+16+1e-4/Sopran/best_model.pth"  # 预训练权重路径

'''
output_dir = r"/home/zx/Valentin_workplace/对比结果"  # 保存输出分数的目录
'''


output_dir = r"/home/zx/Valentin_workplace/Sopran+Mezzo模型输出/1e-5+16+1e-4"


# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# 创建数据集和加载器
class CustomDataset(Dataset):
    def __init__(self, data_dir, train=False, val=False, transforms=None):
        self.data_dir = data_dir
        self.MFCC_data_dir = os.path.join(self.data_dir, 'MFCC_Output')
        self.mfcc_files = [f for f in os.listdir(self.MFCC_data_dir) if f.endswith('_MFCC.xlsx')]
        self.transforms = transforms
        print(f"Found {len(self.mfcc_files)} MFCC files for inference")

    def __len__(self):
        return len(self.mfcc_files)

    def __getitem__(self, idx):
        mfcc_file = self.mfcc_files[idx]
        sample_id = mfcc_file.replace('_MFCC.xlsx', '')
        file_path = os.path.join(self.MFCC_data_dir, mfcc_file)

        # 读取 MFCC 数据
        MFCC_data = pd.read_excel(file_path, header=None, engine="openpyxl").values.astype(float)
        MFCC_tensor = torch.tensor(MFCC_data, dtype=torch.float32).unsqueeze(0)

        if self.transforms is not None:
            MFCC_tensor = self.transforms(MFCC_tensor)

        return MFCC_tensor, sample_id

# 加载验证集
val_dataset = CustomDataset(data_dir, val=True)
print(f"Loaded {len(val_dataset)} samples for inference.")  # 显示加载样本总数
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

# 设备配置（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型
model = CAMPPlus(num_class=num_classes,
                 input_size=1,
                 embd_dim=2560,
                 growth_rate=64,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu').to(device)

# 加载预训练权重
if os.path.exists(pretrained_weights):
    model.load_state_dict(torch.load(pretrained_weights))
    print(f"从 {pretrained_weights} 加载预训练权重")
else:
    raise FileNotFoundError(f"在 {pretrained_weights} 未找到预训练权重")

# 推理函数
def save_predictions_to_excel(net, val_loader, device, output_path):
    net.eval()
    all_preds = []
    all_filenames = []

    with torch.no_grad():
        for im, sample_ids in val_loader:
            im = im.to(device)
            output, _, _ = net(im)
            output = output.view(output.shape[0], 5, 10)  # (B, 5, 10)
            preds = output.argmax(dim=1).cpu().numpy() + 1  # 转为 1-5
            all_preds.append(preds)
            all_filenames.extend(sample_ids)

    # 合并所有 batch
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        print("没有获取到预测数据")
        return

    # 创建 DataFrame
    tech_names = ["Vibrato", "Throat", "Position", "Open", "Clean", "Resonate", "Unify", "Falsetto", "Chest", "Nasal"]
    df_filename = pd.DataFrame({"Filename": all_filenames})
    df_pred = pd.DataFrame(all_preds, columns=[f"Pred_{t}" for t in tech_names])
    df_final = pd.concat([df_filename, df_pred], axis=1)
    df_final = df_final.sort_values(by="Filename")
    df_final.to_excel(output_path, index=False)

    print(f"已保存至 Excel：{output_path}")

# 新增功能：对比预测标签与真实标签
def compare_predictions_with_ground_truth(predictions_path, ground_truth_dir, output_path, txt_path):
    # 读取预测结果
    df_preds = pd.read_excel(predictions_path)
    tech_names = ["Vibrato", "Throat", "Position", "Open", "Clean", "Resonate", "Unify", "Falsetto", "Chest", "Nasal"]
    all_filenames = df_preds["Filename"].tolist()
    pred_array = df_preds[[f"Pred_{t}" for t in tech_names]].values  # 预测标签

    # 存储真实标签
    true_array = []
    missing_files = []
    differences = []

    # 读取真实标签
    for sample_id in all_filenames:
        gt_file = os.path.join(ground_truth_dir, f"{sample_id}.xlsx")
        if os.path.exists(gt_file):
            # 读取 Excel 文件，跳过标题行，提取第二列（评分）
            gt_data = pd.read_excel(gt_file, header=0, engine="openpyxl").iloc[0:11, 1].values
            true_array.append(gt_data)
        else:
            missing_files.append(sample_id)
            true_array.append([0] * 10)  # 填充默认值以保持一致

    true_array = np.array(true_array, dtype=int)

    if missing_files:
        print(f"以下样本的真实标签文件缺失：{missing_files}")

    # 计算指标
    total_samples_per_tech = len(all_filenames) - len(missing_files)
    total_samples_all = total_samples_per_tech * 10  # 总样本数（所有技巧）
    correct_counts = np.sum(pred_array == true_array, axis=0)
    accuracies = correct_counts / total_samples_per_tech if total_samples_per_tech > 0 else np.zeros(10)

    # 计算相差1分、2分、3分及以上的比率
    diff_1_counts = np.sum(np.abs(pred_array - true_array) == 1, axis=0)
    diff_2_counts = np.sum(np.abs(pred_array - true_array) == 2, axis=0)
    diff_3_plus_counts = np.sum(np.abs(pred_array - true_array) >= 3, axis=0)

    diff_1_rates = diff_1_counts / total_samples_per_tech if total_samples_per_tech > 0 else np.zeros(10)
    diff_2_rates = diff_2_counts / total_samples_per_tech if total_samples_per_tech > 0 else np.zeros(10)
    diff_3_plus_rates = diff_3_plus_counts / total_samples_per_tech if total_samples_per_tech > 0 else np.zeros(10)

    # 计算总体指标
    total_correct = correct_counts.sum()
    total_diff_1 = diff_1_counts.sum()
    total_diff_2 = diff_2_counts.sum()
    total_diff_3_plus = diff_3_plus_counts.sum()

    total_accuracy = total_correct / total_samples_all if total_samples_all > 0 else 0
    total_diff_1_rate = total_diff_1 / total_samples_all if total_samples_all > 0 else 0
    total_diff_2_rate = total_diff_2 / total_samples_all if total_samples_all > 0 else 0
    total_diff_3_plus_rate = total_diff_3_plus / total_samples_all if total_samples_all > 0 else 0

    # 控制台打印表格
    print("\n每个技巧的指标（表格格式）：")
    header = ["技巧名称", "准确率", "相差1分率", "相差2分率", "相差3分及以上差异率"]
    # 打印上边框
    print("═" * 100)
    # 打印表头
    print("│ {:<15} │ {:<15} │ {:<15} │ {:<15} │ {:<15} │".format(*header))
    # 打印分隔线
    print("─" * 100)
    # 打印每行数据
    for i, tech in enumerate(tech_names):
        row = [
            tech,
            f"{accuracies[i]:.2%} ({correct_counts[i]}/{total_samples_per_tech})",
            f"{diff_1_rates[i]:.2%} ({diff_1_counts[i]}/{total_samples_per_tech})",
            f"{diff_2_rates[i]:.2%} ({diff_2_counts[i]}/{total_samples_per_tech})",
            f"{diff_3_plus_rates[i]:.2%} ({diff_3_plus_counts[i]}/{total_samples_per_tech})"
        ]
        print("│ {:<18} │ {:<18} │ {:<18} │ {:<18} │ {:<18} │".format(*row))
    # 添加总体行
    total_row = [
        "Total",
        f"{total_accuracy:.2%} ({total_correct}/{total_samples_all})",
        f"{total_diff_1_rate:.2%} ({total_diff_1}/{total_samples_all})",
        f"{total_diff_2_rate:.2%} ({total_diff_2}/{total_samples_all})",
        f"{total_diff_3_plus_rate:.2%} ({total_diff_3_plus}/{total_samples_all})"
    ]
    print("─" * 100)
    print("│ {:<18} │ {:<18} │ {:<18} │ {:<18} │ {:<18} │".format(*total_row))
    # 打印下边框
    print("═" * 100)

    # 查找差异
    for i, sample_id in enumerate(all_filenames):
        for j, tech in enumerate(tech_names):
            pred = pred_array[i, j]
            true = true_array[i, j]
            if pred != true and sample_id not in missing_files:
                differences.append({
                    "Filename": sample_id,
                    "Technique": tech,
                    "Predicted": pred,
                    "Ground Truth": true
                })

    # 保存对比结果到Excel
    df_diff = pd.DataFrame(differences)
    df_acc = pd.DataFrame({
        "技巧名称": tech_names,
        "准确率": [f"{acc:.2%} ({c}/{total_samples_per_tech})" for acc, c in zip(accuracies, correct_counts)],
        "相差1分率": [f"{rate:.2%} ({c}/{total_samples_per_tech})" for rate, c in zip(diff_1_rates, diff_1_counts)],
        "相差2分率": [f"{rate:.2%} ({c}/{total_samples_per_tech})" for rate, c in zip(diff_2_rates, diff_2_counts)],
        "相差3分及以上差异率": [f"{rate:.2%} ({c}/{total_samples_per_tech})" for rate, c in zip(diff_3_plus_rates, diff_3_plus_counts)]
    })
    # 添加总体行到 df_acc
    total_df = pd.DataFrame({
        "技巧名称": ["Total"],
        "准确率": [f"{total_accuracy:.2%} ({total_correct}/{total_samples_all})"],
        "相差1分率": [f"{total_diff_1_rate:.2%} ({total_diff_1}/{total_samples_all})"],
        "相差2分率": [f"{total_diff_2_rate:.2%} ({total_diff_2}/{total_samples_all})"],
        "相差3分及以上差异率": [f"{total_diff_3_plus_rate:.2%} ({total_diff_3_plus}/{total_samples_all})"]
    })
    df_acc = pd.concat([df_acc, total_df], ignore_index=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_acc.to_excel(writer, sheet_name="Accuracy", index=False)
        if not df_diff.empty:
            df_diff.to_excel(writer, sheet_name="Differences", index=False)

    print(f"对比结果已保存至 Excel：{output_path}")

    # 保存到TXT文件（表格格式）
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("每个技巧的指标（表格格式）:\n")
        f.write("═" * 100 + "\n")
        f.write("│ {:<15} │ {:<15} │ {:<15} │ {:<15} │ {:<15} │\n".format(*header))
        f.write("─" * 100 + "\n")
        for i, tech in enumerate(tech_names):
            row = [
                tech,
                f"{accuracies[i]:.2%} ({correct_counts[i]}/{total_samples_per_tech})",
                f"{diff_1_rates[i]:.2%} ({diff_1_counts[i]}/{total_samples_per_tech})",
                f"{diff_2_rates[i]:.2%} ({diff_2_counts[i]}/{total_samples_per_tech})",
                f"{diff_3_plus_rates[i]:.2%} ({diff_3_plus_counts[i]}/{total_samples_per_tech})"
            ]
            f.write("│ {:<18} │ {:<18} │ {:<18} │ {:<18} │ {:<18} │\n".format(*row))
        f.write("─" * 100 + "\n")
        f.write("│ {:<18} │ {:<18} │ {:<18} │ {:<18} │ {:<18} │\n".format(*total_row))
        f.write("═" * 100 + "\n")
    print(f"对比指标已保存至 TXT：{txt_path}")

# 执行推理并保存输出
predictions_path = os.path.join(output_dir, "results.xlsx")
save_predictions_to_excel(
    model,
    val_loader,
    device,
    predictions_path
)

# 执行对比功能
ground_truth_dir = os.path.join(data_dir, "Label")  # 真实标签目录
comparison_excel_path = os.path.join(output_dir, "comparison_results.xlsx")
comparison_txt_path = os.path.join(output_dir, "comparison_results.txt")  # TXT路径
compare_predictions_with_ground_truth(
    predictions_path,
    ground_truth_dir,
    comparison_excel_path,
    comparison_txt_path
)