#!/usr/bin/env python3
"""
音频拆分脚本 - 本地调试版本
将大音频文件按10秒一段进行拆分，用于在线评分任务批量创建

命令行参数：
--input: 输入音频文件路径
--singer: 歌手名
--song: 歌曲名
--part: 声部（sopran/mezzo/falsetto/tenor/baritone/bass）
--output_dir: 输出目录

输出文件命名格式：{歌手名}-{歌曲名}-{声部}_001.wav, {歌手名}-{歌曲名}-{声部}_002.wav, ...
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import json

# 声部类型映射
VOICE_TYPES = {
    'sopran': '女高音',
    'mezzo': '女中音', 
    'falsetto': '假声',
    'tenor': '男高音',
    'baritone': '男中音',
    'bass': '男低音'
}

def check_ffmpeg():
    """检查FFmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10, encoding='utf-8', errors='ignore')
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def get_audio_duration(input_file):
    """获取音频文件时长（秒）"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
        else:
            print(f"❌ 获取音频时长失败: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ 获取音频时长异常: {e}")
        return None

def split_audio(input_file, output_dir, singer, song, part, segment_duration=10):
    """
    拆分音频文件
    
    Args:
        input_file: 输入音频文件路径
        output_dir: 输出目录
        singer: 歌手名
        song: 歌曲名
        part: 声部
        segment_duration: 每段时长（秒），默认10秒
    
    Returns:
        List[str]: 生成的音频文件路径列表
    """
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return []
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取音频时长
    total_duration = get_audio_duration(input_file)
    if total_duration is None:
        print("❌ 无法获取音频时长，拆分失败")
        return []
    
    print(f"📊 音频总时长: {total_duration:.2f}秒")
    
    # 计算需要拆分的段数
    num_segments = int(total_duration / segment_duration) + (1 if total_duration % segment_duration > 0 else 0)
    print(f"📊 预计拆分段数: {num_segments}")
    
    generated_files = []
    
    for i in range(num_segments):
        start_time = i * segment_duration
        
        # 计算当前段的实际时长
        remaining_duration = total_duration - start_time
        current_duration = min(segment_duration, remaining_duration)
        
        # 如果剩余时长太短（小于1秒），跳过
        if current_duration < 1.0:
            print(f"⏭️ 跳过过短片段 {i+1} (时长: {current_duration:.2f}秒)")
            continue
        
        # 生成输出文件名
        output_filename = f"{singer}-{song}-{part}_{i+1:03d}.wav"
        output_file = output_path / output_filename
        
        # FFmpeg命令
        cmd = [
            'ffmpeg', '-y',  # -y 覆盖输出文件
            '-i', input_file,
            '-ss', str(start_time),  # 开始时间
            '-t', str(current_duration),  # 持续时间
            '-acodec', 'pcm_s16le',  # 音频编码
            '-ar', '44100',  # 采样率
            '-ac', '2',  # 声道数
            str(output_file)
        ]
        
        try:
            print(f"🔄 拆分片段 {i+1}/{num_segments}: {start_time:.1f}s - {start_time + current_duration:.1f}s")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                if output_file.exists() and output_file.stat().st_size > 0:
                    generated_files.append(str(output_file))
                    print(f"✅ 生成文件: {output_filename}")
                else:
                    print(f"❌ 文件生成失败: {output_filename}")
            else:
                print(f"❌ FFmpeg错误: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ 拆分超时: {output_filename}")
        except Exception as e:
            print(f"❌ 拆分异常: {e}")
    
    print(f"🎉 拆分完成，共生成 {len(generated_files)} 个文件")
    return generated_files

def main():
    parser = argparse.ArgumentParser(description='音频拆分脚本 - 本地调试版本')
    parser.add_argument('--input', required=True, help='输入音频文件路径')
    parser.add_argument('--singer', required=True, help='歌手名')
    parser.add_argument('--song', required=True, help='歌曲名')
    parser.add_argument('--part', required=True, choices=list(VOICE_TYPES.keys()), 
                       help='声部类型')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--duration', type=int, default=10, help='每段时长（秒），默认10秒')
    
    args = parser.parse_args()
    
    print("🎵 音频拆分脚本启动")
    print(f"📁 输入文件: {args.input}")
    print(f"🎤 歌手: {args.singer}")
    print(f"🎵 歌曲: {args.song}")
    print(f"🎼 声部: {args.part} ({VOICE_TYPES.get(args.part, '未知')})")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"⏱️ 每段时长: {args.duration}秒")
    
    # 检查FFmpeg
    if not check_ffmpeg():
        print("❌ FFmpeg不可用，请确保已安装FFmpeg并添加到PATH")
        sys.exit(1)
    
    # 执行拆分
    generated_files = split_audio(
        args.input, 
        args.output_dir, 
        args.singer, 
        args.song, 
        args.part,
        args.duration
    )
    
    if generated_files:
        print(f"✅ 拆分成功，生成了 {len(generated_files)} 个文件:")
        for file_path in generated_files:
            print(f"  - {os.path.basename(file_path)}")
        sys.exit(0)
    else:
        print("❌ 拆分失败，没有生成任何文件")
        sys.exit(1)

if __name__ == '__main__':
    main()
