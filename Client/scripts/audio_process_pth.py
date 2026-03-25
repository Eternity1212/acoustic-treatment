import os
import subprocess
import time
import argparse
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import detect_nonsilent

import torch

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"UVR 使用设备: {device}")
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='音频处理脚本（仅保留最终切割片段）')
    parser.add_argument('--input', required=True, help='输入音频文件路径')
    parser.add_argument('--singer', required=True, help='歌手名')
    parser.add_argument('--song', required=True, help='歌曲名')
    parser.add_argument('--part', required=True, help='声部')
    parser.add_argument('--output_dir', required=True, help='输出根目录（客户端已提前创建）')
    args = parser.parse_args()

    # 音频处理参数
    silence_threshold = -40  # 静音阈值（dB）
    seek_step = 100  # 检测步长（毫秒）
    min_silence_len = 3 * 1000  # 最小静音长度（3秒）
    chunk_length_ms = 10 * 1000  # 每个片段长度（10秒）

    # 直接使用客户端创建的输出目录
    final_output_folder = args.output_dir

    # 临时文件路径（不保存，仅在内存中处理）
    input_basename = os.path.basename(args.input)
    input_name = os.path.splitext(input_basename)[0]

    try:
        print(f"开始用 UVR CLI 处理 {input_basename} ...")

        # 调用 UVR CLI（输出到客户端创建的目录）
        # 调用 UVR CLI（添加工作目录参数 cwd）
        uvr_command = [
            "python", "-u", r"D:\competition\separate_diva_pth.py",
            "--input", args.input,
            "--output", final_output_folder
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            uvr_command,
            check=True,
            env=env,
            timeout=1800,
        )

        # 定义UVR生成的中间文件路径
        vocals_file = os.path.join(final_output_folder, f"{input_name}_Vocals.wav")
        novocals_file = os.path.join(final_output_folder, f"{input_name}_NoVocals.wav")

        # 检查人声文件是否存在
        if not os.path.exists(vocals_file):
            raise FileNotFoundError(f"未找到分离后的人声文件: {vocals_file}")

        # 读取人声文件（仅在内存中处理，不保存到磁盘）
        audio = AudioSegment.from_file(vocals_file, format="wav")

        # 检测并处理非静音片段（仅在内存中操作）
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_threshold,
            seek_step=seek_step
        )

        processed_audio = AudioSegment.empty()
        if nonsilent_ranges:
            for start, end in nonsilent_ranges:
                processed_audio += audio[start:end]
            print(f"处理前长度: {len(audio) / 1000:.2f}秒")
            print(f"处理后长度: {len(processed_audio) / 1000:.2f}秒")
        else:
            processed_audio = audio
            print("未检测到非静音片段，使用原始音频")

        # 切割音频并保存最终片段（保存到客户端创建的目录）
        chunks = make_chunks(processed_audio, chunk_length_ms)
        chunk_count = 0
        for i, chunk in enumerate(chunks):
            if len(chunk) >= chunk_length_ms / 2:
                chunk_name = f"{args.singer}-{args.song}-{args.part}-{i + 1:03d}.wav"
                result_path = os.path.join(final_output_folder, chunk_name)
                chunk.export(result_path, format="wav")  # 只保存最终切割片段
                chunk_count += 1

        # # 彻底删除所有中间文件（UVR生成的人声和非人声文件）
        # for file_path in [vocals_file, novocals_file]:
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        #         print(f"已删除中间文件: {os.path.basename(file_path)}")

        print(f"成功切割为 {chunk_count} 个片段")
        print(f"输出目录: {final_output_folder}")

    except subprocess.CalledProcessError as e:
        print(f"UVR CLI 处理失败: {e}")
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


if __name__ == "__main__":
    main()