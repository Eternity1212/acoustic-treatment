import numpy as np
import librosa
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
from CAM_S import CAMPPlus  # 请确保正确导入此模块
import argparse

# === 预定义的权重文件路径（请根据需要修改） ===
WEIGHT_FILES = {
    "sopran": r"/home/zx/AST1/logs_ddnet_sopran_2637/2025-08-28_17-36-59/best_model.pth",       # Part1 女高
    "mezzo": r"/home/zx/AST1/logs_ddnet_mezzo_1024/2025-08-28_19-06-05/best_model.pth",      # Part2 女中
    "tenor": r"/home/zx/AST1/logs_ddnet_tenor_1335/2025-08-28_19-45-18/best_model.pth",      # Part4 男高
    "baritone": r"/home/zx/AST1/logs_ddnet_baritone/2025-07-29_12-04-20/best_model.pth", # Part5 男中
    # 作为占位，日后更新模型时，请将此处的权重文件路径替换为对应实际路径
    "faltatto": r"/home/zx/AST1/logs_ddnet_mezzo/2025-07-21_11-47-08/best_model.pth",   # Part3 假声男高
    "bass": r"/home/zx/AST1/logs_ddnet_bass/2025-07-29_12-04-20/best_model.pth",        # Part6 男低
}

# === 超参数 ===
TARGET_SR = 48000  # MFCC提取的目标采样率
MAX_PAD_LEN = 128  # MFCC特征的最大填充长度
VAL_BATCH_SIZE = 16  # 推理的批量大小
NUM_WORKERS = 4  # 数据加载的线程数
NUM_CLASSES = 50  # 模型的类别数
VOICE_PARTS = ['sopran', 'mezzo', 'tenor', 'baritone', 'faltatto', 'bass']  # 支持的声部


# === 提取MFCC特征的函数 ===
def extract_mfcc_features(file_path, max_pad_len=MAX_PAD_LEN, target_sr=TARGET_SR):
    try:
        # 使用目标采样率加载音频并计算MFCC
        audio, sample_rate = librosa.load(file_path, sr=target_sr, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
        # 填充或截断MFCC特征
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width < 0:
            mfccs = mfccs[:, :max_pad_len]
        else:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print(f"处理文件时发生错误: {file_path}, 异常: {e}")
        mfccs = None
    return mfccs


# === 自定义推理数据集 ===
class CustomDataset(Dataset):
    def __init__(self, mfcc_file, transforms=None):
        self.mfcc_file = mfcc_file
        self.transforms = transforms
        print(f"加载MFCC文件: {mfcc_file}")

    def __len__(self):
        return 1  # 仅处理单个文件

    def __getitem__(self, idx):
        sample_id = os.path.splitext(os.path.basename(self.mfcc_file))[0].replace('_MFCC', '')
        # 读取MFCC数据
        mfcc_data = pd.read_excel(self.mfcc_file, header=None, engine="openpyxl").values.astype(float)
        mfcc_tensor = torch.tensor(mfcc_data, dtype=torch.float32).unsqueeze(0)
        if self.transforms is not None:
            mfcc_tensor = self.transforms(mfcc_tensor)
        return mfcc_tensor, sample_id


# === 推理并保存结果到Excel的函数 ===
def save_predictions_to_excel(net, val_loader, device, output_path):
    net.eval()
    all_preds = []
    all_filenames = []

    with torch.no_grad():
        for im, sample_ids in val_loader:
            im = im.to(device)
            output, _, _ = net(im)
            output = output.view(output.shape[0], 5, 10)  # (B, 5, 10)
            preds = output.argmax(dim=1).cpu().numpy() + 1  # 转为1-5
            all_preds.append(preds)
            all_filenames.extend(sample_ids)

    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        print("没有获取到预测数据")
        return

    # 类别标签
    class_labels = [
        "Vibrato", "Throat", "Position", "Open", "Clean", "Resonate", "Unify", "Falsetto", "Chest", "Nasal"
    ]

    # 我们假设每个文件只有一个预测（例如一段音频对应一个预测结果）
    results = []

    for idx, sample_id in enumerate(all_filenames):
        preds = all_preds[idx]
        result = list(zip(class_labels, preds))
        results.extend(result)

    # 创建一个 DataFrame 来保存结果
    df = pd.DataFrame(results, columns=["Class", "Value"])

    # 保存为Excel
    df.to_excel(output_path, index=False)
    print(f"预测结果已保存至: {output_path}")


# === 解析命令行参数的函数 ===
def parse_arguments():
    parser = argparse.ArgumentParser(description="音频处理与推理程序")
    parser.add_argument('--audiofile', type=str, required=True, help="输入的WAV音频文件路径")
    parser.add_argument('--mffcdir', type=str, required=True, help="MFCC输出目录路径")
    parser.add_argument('--outputdir', type=str, required=True, help="推理打分结果目录路径")
    parser.add_argument('--part', type=str, required=True, choices=VOICE_PARTS,
                        help="声部选择 (sopran/mezzo/tenor/baritone)")
    args = parser.parse_args()

    # 验证音频文件
    if not (os.path.exists(args.audiofile) and args.audiofile.endswith('.wav')):
        raise ValueError("音频文件不存在或不是WAV文件！")

    # 生成推理打分输出文件名
    base_name = os.path.splitext(os.path.basename(args.audiofile))[0]
    output_scores_file = os.path.join(args.outputdir, "predictions.xlsx")  #推理打分输出文件名修改

    return args.audiofile, args.mffcdir, output_scores_file, args.part


# === 主程序 ===
def main():
    # 步骤1：解析命令行参数
    print("\n开始音频处理和推理程序...")
    input_audio_file, output_mfcc_dir, output_scores_file, voice_part = parse_arguments()
    print(f"\n检测到的声部: {voice_part.capitalize()}")
    print(f"输入音频文件: {input_audio_file}")
    print(f"MFCC输出目录: {output_mfcc_dir}")
    print(f"推理打分结果文件: {output_scores_file}")

    # 步骤2：加载对应声部的预训练权重
    pretrained_weights = WEIGHT_FILES.get(voice_part)
    if not pretrained_weights or not os.path.exists(pretrained_weights):
        print(f"未找到 {voice_part} 的预训练权重文件: {pretrained_weights}")
        return
    print(f"正在加载 {voice_part} 的预训练权重: {pretrained_weights}")

    # 步骤3：提取MFCC特征
    print("\n开始提取MFCC特征...")
    if not os.path.exists(output_mfcc_dir):
        os.makedirs(output_mfcc_dir)

    mfccs = extract_mfcc_features(input_audio_file, target_sr=TARGET_SR)
    if mfccs is not None:
        # 保存MFCC到Excel
        filename = os.path.splitext(os.path.basename(input_audio_file))[0] + "_MFCC.xlsx"
        filepath = os.path.join(output_mfcc_dir, filename)
        df = pd.DataFrame(mfccs)
        df.to_excel(filepath, index=False, header=False)
        print(f"MFCC特征已保存至: {filepath}")
    else:
        print("MFCC提取失败")
        return

    # 步骤4：初始化模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(
        num_class=NUM_CLASSES,
        input_size=1,
        embd_dim=8192,
        growth_rate=64,
        bn_size=4,
        init_channels=128,
        config_str='batchnorm-relu'
    ).to(device)

    model.load_state_dict(torch.load(pretrained_weights))
    print(f"已加载 {voice_part} 的模型权重")

    # 步骤5：创建推理数据集和数据加载器
    print("\n正在准备推理数据...")
    val_dataset = CustomDataset(filepath)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 步骤6：执行推理并保存结果
    print("\n开始推理...")
    os.makedirs(os.path.dirname(output_scores_file), exist_ok=True)
    save_predictions_to_excel(model, val_loader, device, output_scores_file)

    # 步骤7：完成提示
    print("\n代码运行完成！")


if __name__ == "__main__":
    main()
