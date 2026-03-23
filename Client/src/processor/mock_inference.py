#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟推理脚本 - 用于Client本地调试模式
模仿 model_scoring_client/scripts/inference_scores.py 的功能
"""

import argparse
import json
import time
import random
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# 声部类型映射
VOICE_PARTS = {
    'sopran': '女高',
    'mezzo': '女中',
    'falsetto': '假声男高',
    'tenor': '男高',
    'baritone': '男中',
    'bass': '男低'
}

def validate_voice_part(part):
    """验证声部类型参数"""
    if not part:
        return None

    part_lower = part.lower()
    if part_lower in VOICE_PARTS:
        return part_lower

    print(f"警告: 未知的声部类型 '{part}'")
    print(f"支持的声部类型: {', '.join(VOICE_PARTS.keys())}")
    return None

def simulate_audio_analysis(audio_file, mfcc_dir, output_dir, voice_part):
    """模拟音频分析过程"""
    print(f"正在分析音频文件: {audio_file}")
    print(f"声部类型: {voice_part} ({VOICE_PARTS[voice_part]})")
    print(f"MFCC输出目录: {mfcc_dir}")
    print(f"结果输出目录: {output_dir}")

    # 检查音频文件是否存在（调试模式下可能不存在）
    audio_path = Path(audio_file)
    if audio_path.exists():
        print(f"检测到音频文件: {audio_path.name}, 大小: {audio_path.stat().st_size} bytes")
    else:
        print(f"调试模式: 使用模拟音频文件路径 {audio_file}")
    
    # 模拟加载音频
    print("加载音频文件...")
    time.sleep(0.5)

    # 模拟MFCC特征提取
    print("提取MFCC特征...")
    print(f"   - 保存MFCC到: {mfcc_dir}")
    time.sleep(0.3)

    # 模拟根据声部选择模型权重
    print(f"根据声部 {voice_part} 自动选择模型权重...")
    model_weights = {
        'sopran': 'models/sopran_model.pth',
        'mezzo': 'models/mezzo_model.pth',
        'falsetto': 'models/falsetto_model.pth',
        'tenor': 'models/tenor_model.pth',
        'baritone': 'models/baritone_model.pth',
        'bass': 'models/bass_model.pth'
    }
    selected_model = model_weights.get(voice_part, 'models/default_model.pth')
    print(f"   - 使用模型: {selected_model}")
    time.sleep(0.2)

    # 模拟其他特征提取
    print("提取其他音频特征...")
    print("   - 频谱质心计算...")
    time.sleep(0.2)
    print("   - 谐波比率分析...")
    time.sleep(0.4)

    # 根据声部类型调整分析策略
    print(f"针对{voice_part}声部进行专门分析...")
    time.sleep(0.3)

    # 模拟模型推理
    print("执行模型推理...")
    time.sleep(0.8)

    # 生成随机结果
    has_accompaniment = random.choice([True, False])
    confidence = random.uniform(0.7, 0.95)

    return has_accompaniment, confidence

def create_excel_output(excel_file, has_accompaniment, confidence):
    """创建Excel输出文件，格式与批量上传数据集一致"""
    try:
        # 模拟技能标签和分数数据（使用前端支持的技能标签）
        available_skills = ["Vibrato", "Throat", "Position", "Open", "Resonate", "Unify", "Falsetto", "Chest", "Nasal", "Clean"]

        # 随机选择4-6个技能进行评分
        num_skills = random.randint(4, 6)
        selected_skills = random.sample(available_skills, num_skills)

        skills_data = []

        # 根据伴奏检测结果生成不同的技能评分
        if has_accompaniment:
            # 有伴奏的情况 - 各项技能分数相对较低
            for skill in selected_skills:
                score = random.randint(2, 4)
                skills_data.append([skill, score])
        else:
            # 无伴奏的情况 - 各项技能分数相对较高
            for skill in selected_skills:
                score = random.randint(3, 5)
                skills_data.append([skill, score])

        # 根据置信度调整分数
        confidence_factor = confidence
        adjusted_skills_data = []
        for skill, score in skills_data:
            # 置信度越高，分数越接近原值；置信度低则增加随机性
            if confidence_factor > 0.8:
                adjusted_score = score
            else:
                # 低置信度时增加±1的随机变化
                adjustment = random.choice([-1, 0, 1])
                adjusted_score = max(1, min(5, score + adjustment))

            adjusted_skills_data.append([skill, adjusted_score])

        # 创建DataFrame（使用标准列名）
        df = pd.DataFrame(adjusted_skills_data, columns=['Skill Tag', 'Score'])

        # 保存为Excel文件
        df.to_excel(excel_file, index=False, engine='openpyxl')

        print(f"Excel结果文件已保存: {excel_file}")
        print(f"技能评分详情:")
        for skill, score in adjusted_skills_data:
            print(f"   {skill}: {score}分")

        return True

    except Exception as e:
        print(f"创建Excel文件失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频伴奏检测推理脚本')
    parser.add_argument('--audiofile', required=True, help='音频文件路径')
    parser.add_argument('--mffcdir', required=True, help='MFCC输出路径')
    parser.add_argument('--outputdir', required=True, help='推理打分输出路径')
    parser.add_argument('--part', required=True, help='声部类型 (sopran/mezzo/falsetto/tenor/baritone/bass)')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 验证声部类型
    validated_part = validate_voice_part(args.part)
    if not validated_part:
        print(f"错误: 无效的声部类型 '{args.part}'")
        print(f"支持的声部类型: {', '.join(VOICE_PARTS.keys())}")
        return 1

    part_display = f"{validated_part} ({VOICE_PARTS[validated_part]})"

    print("音频伴奏检测推理脚本")
    print("=" * 50)
    print(f"音频文件: {args.audiofile}")
    print(f"MFCC输出: {args.mffcdir}")
    print(f"输出目录: {args.outputdir}")
    print(f"声部类型: {part_display}")
    print("=" * 50)
    
    # 验证输入文件
    audio_path = Path(args.audiofile)
    if not audio_path.exists():
        print(f"错误: 音频文件不存在 - {args.audiofile}")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建MFCC输出目录
    mfcc_dir = Path(args.mffcdir)
    mfcc_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 执行音频分析
        has_accompaniment, confidence = simulate_audio_analysis(
            args.audiofile, args.mffcdir, args.outputdir, validated_part
        )
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 生成详细结果
        result = {
            "has_accompaniment": has_accompaniment,
            "confidence": round(confidence, 3),
            "processed_at": datetime.now().isoformat(),
            "input_file": str(args.audiofile),
            "voice_part": validated_part,
            "voice_part_name": VOICE_PARTS[validated_part],
            "mfcc_output_dir": str(args.mffcdir),
            "processing_details": {
                "duration": round(random.uniform(5, 10), 2),
                "sample_rate": 44100,
                "channels": random.choice([1, 2]),
                "file_size_mb": round(audio_path.stat().st_size / (1024*1024), 2),
                "features_extracted": [
                    "mfcc",
                    "spectral_centroid",
                    "spectral_rolloff",
                    "zero_crossing_rate",
                    "harmonic_ratio"
                ],
                "algorithm": f"accompaniment_detection_{validated_part}_v1.0",
                "model_version": "1.0.0",
                "processing_time_seconds": round(processing_time, 3),
                "auto_selected_model": f"models/{validated_part}_model.pth"
            },
            "metadata": {
                "script_version": "1.0.0",
                "python_version": sys.version,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # 保存JSON结果文件（保持兼容性）
        result_file = output_dir / "result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 生成Excel结果文件（主要输出）
        excel_file = output_dir / "scores.xlsx"
        create_excel_output(excel_file, has_accompaniment, confidence)
        
        # 输出结果
        print("\n处理完成!")
        print(f"检测结果: {'有伴奏' if has_accompaniment else '无伴奏'}")
        print(f"置信度: {confidence:.3f}")
        print(f"处理时间: {processing_time:.3f} 秒")
        print(f"JSON结果已保存到: {result_file}")
        print(f"Excel结果已保存到: {excel_file}")

        if args.verbose:
            print(f"\n详细结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        print("\n推理脚本执行成功!")
        
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        
        # 保存错误信息
        error_file = output_dir / "error.json"
        error_data = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "input_file": str(args.audiofile),
            "voice_part": validated_part
        }
        
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        
        print(f"错误信息已保存到: {error_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()
