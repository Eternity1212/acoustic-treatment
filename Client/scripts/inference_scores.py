import os
import torch
import pandas as pd
import numpy as np
import librosa
import librosa.display
import argparse
from torch.utils.data import DataLoader, Dataset
from CAM import CAMPPlus  # 根据你的实际模块导入

# 定义将.Wav文件转化为MFCC表格的函数
def extract_mfcc_features(file_path, max_pad_len=128):
    try:
        # 使用librosa加载音频文件，并计算MFCC特征
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', sr=48000)
        print(f"音频文件 '{os.path.basename(file_path)}' 的采样率为: {sample_rate} Hz")

        mfccs = librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=128)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width < 0:
            mfccs = mfccs[:, :max_pad_len]
        else:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        print("Error encountered while parsing file: ", file_path, "\nException:", e)
        mfccs = None
    return mfccs

# 数据集类
class CustomDataset(Dataset):
    def __init__(self, mfcc_data_dir, transforms=None):
        self.MFCC_data_dir = mfcc_data_dir
        self.mfcc_files = [f for f in os.listdir(self.MFCC_data_dir) if f.endswith('_MFCC.xlsx')]
        self.transforms = transforms
        print(f"Found {len(self.mfcc_files)} MFCC files for inference")

    def __len__(self):
        return len(self.mfcc_files)

    def __getitem__(self, idx):
        mfcc_file = self.mfcc_files[idx]
        sample_id = mfcc_file.replace('_MFCC.xlsx', '')
        file_path = os.path.join(self.MFCC_data_dir, mfcc_file)

        MFCC_data = pd.read_excel(file_path, header=None, engine="openpyxl").values.astype(float)
        MFCC_tensor = torch.tensor(MFCC_data, dtype=torch.float32).unsqueeze(0)

        if self.transforms is not None:
            MFCC_tensor = self.transforms(MFCC_tensor)

        return MFCC_tensor, sample_id

# 推理函数
def save_predictions_to_excel(model, val_loader, device, output_path):
    model.eval()
    # 定义技巧名称映射表
    TECHNIQUE_ORDER = [
        'Vibrato',
        'Throat',
        'Position',
        'Open',
        'Clean',
        'Resonate',
        'Unify',
        'Falsetto',
        'Chest',
        'Nasal'
    ]

    all_results = []  # 存储所有结果

    with torch.no_grad():
        for im, sample_ids in val_loader:
            im = im.to(device)
            output, _, _ = model(im)
            output = output.view(output.shape[0], 5, 10)  # (B, 5, 10)
            preds = output.argmax(dim=1).cpu().numpy() + 1  # 转为 1-5

            for i, sample_id in enumerate(sample_ids):
                # 为每个样本创建表格形式的数据
                file_results = []
                for j, technique in enumerate(TECHNIQUE_ORDER):
                    file_results.append({
                        'Class': technique,
                        'Value': preds[i, j]
                    })

                all_results.append({
                    'Filename': sample_id,
                    'Predictions': file_results
                })

    # 创建最终Excel文件
    with pd.ExcelWriter(output_path) as writer:
        for file_info in all_results:
            filename = file_info['Filename']
            predictions = file_info['Predictions']

            # 创建包含两列的DataFrame
            df = pd.DataFrame({
                'Class': [item['Class'] for item in predictions],
                'Value': [item['Value'] for item in predictions]
            })

            # 写入工作表，保持原始技巧顺序
            df.to_excel(
                writer,
                sheet_name=filename[:31],  # 工作表名最长31字符
                index=False,
                header=['Class', 'Value']  # 设置列名
            )

    print(f"✅ 预测结果已保存至Excel: {output_path}")

# 主函数
def main(audio_file, model_weight, output_dir):
    # 生成MFCC特征
    mfccs = extract_mfcc_features(audio_file)
    if mfccs is None:
        print("音频文件处理失败")
        return

    # 保存MFCC特征到Excel
    mfcc_filename = os.path.basename(audio_file).replace('.wav', '_MFCC.xlsx')
    mfcc_filepath = os.path.join(output_dir, mfcc_filename)
    df = pd.DataFrame(mfccs)
    df.to_excel(mfcc_filepath, index=False, header=False)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(num_class=50, input_size=1, embd_dim=8192, growth_rate=64, bn_size=4, init_channels=128).to(device)
    if os.path.exists(model_weight):
        model.load_state_dict(torch.load(model_weight))
        print(f"Loaded pretrained weights from {model_weight}")
    else:
        raise FileNotFoundError(f"Pretrained weights not found at {model_weight}")

    # 创建数据集和加载器
    dataset = CustomDataset(output_dir)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 执行推理并保存输出
    save_predictions_to_excel(model, val_loader, device, os.path.join(output_dir, "predictions.xlsx"))

# 命令行解析
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Deployment for Audio Classification")
    parser.add_argument('--audio_file', type=str, required=True, help="Path to the audio file")
    parser.add_argument('--model_weight', type=str, required=True, help="Path to the pretrained model weights")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output files")
    args = parser.parse_args()

    main(args.audio_file, args.model_weight, args.output_dir)
