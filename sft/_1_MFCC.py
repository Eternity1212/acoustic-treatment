import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pandas as pd
import os


# 定义将.Wav文件转化为MFCC表格的函数
def extract_mfcc_features(file_path, max_pad_len=128):
    try:
        # 使用librosa加载音频文件，并计算MFCC特征
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast',sr=None)
        #print(f"音频文件 '{os.path.basename(file_path)}' 的采样率为: {sample_rate} Hz")
        audio = librosa.util.normalize(audio)

        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        # pad_width = max_pad_len - mfccs.shape[1]
        # mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        # 如果MFCC特征的长度小于max_pad_len，则填充；如果大于，则截断
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width < 0:
            # 处理pad_width为负数的情况，比如通过截断mfccs到max_pad_len长度
            mfccs = mfccs[:, :max_pad_len]
        else:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        # 如果处理过程中出现异常，打印错误信息
        print("Error encountered while parsing file: ", file_path, "\nException:", e)
        mfccs = None
    return mfccs


# 将所有WAV转化为MFCC表格
directory = r"/home/zx/Valentin_workplace/Mezzo_test"
Audio_path = os.path.join(directory, "Audio")

files = os.listdir(Audio_path)

if not os.path.exists(os.path.join(directory, "MFCC_Output")):
    os.makedirs(os.path.join(directory, "MFCC_Output"))

# Loop through each file
for file in files:
    # Check if the file is an Excel file
    if file.endswith('.wav'):
        mfccs = extract_mfcc_features(os.path.join(Audio_path, file))
        # df = pd.DataFrame(mfccs)
        filename = file.split('.')[0] + "_MFCC.xlsx"
        filepath = os.path.join(directory, "MFCC_Output", filename)

        df = pd.DataFrame(mfccs)
        df.to_excel(filepath, index=False, header=False)


































