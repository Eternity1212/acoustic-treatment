# 01_extract_mfcc.py
import numpy as np
import librosa
import pandas as pd
import os
import torch

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

# 可选：保存重采样音频
SAVE_RESAMPLED_WAV = False

def extract_mfcc_features(file_path, max_pad_len=128, target_sr=44100, n_mfcc=128):
    try:
        audio_raw, original_sr = librosa.load(file_path, sr=None, mono=True)
        if original_sr != target_sr:
            audio = librosa.resample(audio_raw, orig_sr=original_sr, target_sr=target_sr)
        else:
            audio = audio_raw

        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.astype(np.float32), original_sr, target_sr, audio

    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return None, None, None, None

# ============ 路径设置 ============
directory = r"D:\比赛视频Videos\sopran_cutted"
Audio_path = os.path.join(directory, "Audio")
MFCC_out = os.path.join(directory, "MFCC_Output")
os.makedirs(MFCC_out, exist_ok=True)

Resampled_out = os.path.join(directory, "Audio_44100")
if SAVE_RESAMPLED_WAV and HAS_SF:
    os.makedirs(Resampled_out, exist_ok=True)

files = os.listdir(Audio_path)
wav_files = [f for f in files if f.lower().endswith(".wav")]

print(f"📁 输入目录: {Audio_path}")
print(f"📦 共检测到 WAV 文件: {len(wav_files)}")
print(f"🧾 MFCC 输出目录: {MFCC_out}")

# ============ 主循环 ============
for idx, file in enumerate(wav_files, start=1):
    in_path = os.path.join(Audio_path, file)

    mfccs, original_sr, target_sr, audio_44100 = extract_mfcc_features(
        in_path, max_pad_len=128, target_sr=44100, n_mfcc=128
    )

    if mfccs is None:
        continue

    # 保存为 .pt 文件（推荐，便于后续加载）
    out_name = os.path.splitext(file)[0] + "_MFCC.pt"
    out_path = os.path.join(MFCC_out, out_name)
    torch.save(torch.tensor(mfccs).unsqueeze0), out_path) ( # (1, 128, 128)

    # 可选：保存重采样音频
    if SAVE_RESAMPLED_WAV and HAS_SF:
        wav_out_path = os.path.join(Resampled_out, file)
        sf.write(wav_out_path, audio_44100, target_sr, subtype="PCM_16")

    print(f"✅ [{idx}/{len(wav_files)}] {file} -> MFCC saved: {out_path}")

print("🎉 MFCC 提取完成。")