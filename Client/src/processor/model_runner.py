"""
模型执行器模块
"""

import asyncio
import subprocess
import tempfile
import os
import json
import sys
from typing import Dict, Any, Optional

# 添加src目录到Python路径
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_dir)

from pathlib import Path
from config.config import MODEL_SCORING_SCRIPT_DEBUG, MODEL_SCORING_SCRIPT_PROD, IS_DEBUG
from utils.logger import get_logger
from utils.file_utils import ensure_dir, create_temp_file, safe_remove_file

logger = get_logger("model_runner")

class ModelRunner:
    """模型执行器，负责调用评分脚本进行模型推理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 根据调试模式选择脚本路径
        self.script_path = MODEL_SCORING_SCRIPT_DEBUG if IS_DEBUG else MODEL_SCORING_SCRIPT_PROD
        self.timeout = config.get('model_timeout', 300)  # 5分钟超时
        
        # 确保输出目录存在
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"模型执行器初始化完成，脚本路径: {self.script_path}")
    
    async def run_scoring(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        执行模型评分
        
        Args:
            task: 任务信息，包含音频文件路径等
            
        Returns:
            Optional[Dict]: 评分结果，失败返回None
        """
        task_id = task.get('result_id') or task.get('task_id')
        audio_file = task.get('audio_file_path')
        voice_type = task.get('voice_type', 'sopran')  # 默认女高
        
        if not audio_file:
            logger.error(f"任务 {task_id} 缺少音频文件路径")
            return None
        
        try:
            logger.info(f"开始执行模型评分: 任务={task_id}, 文件={audio_file}")
            
            # 统一使用模拟推理方法，根据IS_DEBUG标志选择不同的脚本
            return await self._run_mock_inference(task_id, audio_file, voice_type)

        except Exception as e:
            logger.error(f"执行模型评分异常: 任务={task_id}, 错误={e}")
            return None

    async def _log_stream(self, stream, is_error: bool = False):
        """实时读取子进程输出并打印到日志"""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                msg = line.decode('utf-8', errors='ignore').rstrip()
                if msg:
                    if is_error:
                        logger.error(f"[子进程] {msg}")
                    else:
                        logger.info(f"[子进程] {msg}")
        except Exception as e:
            logger.warning(f"读取子进程日志失败: {e}")

    async def _run_mock_inference(self, task_id: str, audio_file: str, voice_type: str) -> Optional[Dict[str, Any]]:
        """
        运行推理（根据IS_DEBUG标志选择模拟或真实推理脚本）
        
        Args:
            task_id: 任务ID
            audio_file: 音频文件路径
            voice_type: 声部类型
            
        Returns:
            Optional[Dict]: 推理结果
        """
        try:

            # 创建任务专用输出目录
            task_output_dir =self.output_dir / f"task_{task_id}"
            task_output_dir.mkdir(exist_ok=True)
            
            # 创建MFCC输出目录
            mfcc_dir = task_output_dir / "mfcc"
            mfcc_dir.mkdir(exist_ok=True)
            

            # 使用真实推理脚本
            script_path = Path(self.script_path)

            logger.info("="*20)
            logger.info(f"开始执行推理: 任务={task_id}, 音频文件={audio_file}, 声部={voice_type}")
            logger.info('脚本路径: {script_path}')
            
            # 构建命令参数
            cmd = [
                sys.executable,
                str(script_path),
                '--audiofile', audio_file,
                '--mfccdir', str(mfcc_dir),
                '--outputdir', str(task_output_dir),
                '--part', voice_type
            ]
            
            logger.info(f"执行命令: {cmd}")
            logger.info("="*20)
            
            # 异步执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                # 实时输出子进程日志
                stdout_task = asyncio.create_task(self._log_stream(process.stdout, is_error=False))
                stderr_task = asyncio.create_task(self._log_stream(process.stderr, is_error=True))

                # 等待执行完成（带超时）
                await asyncio.wait_for(process.wait(), timeout=self.timeout)
                # 确保日志读取完成
                await stdout_task
                await stderr_task
                
                if process.returncode == 0:
                    # 读取结果文件
                    result_file = task_output_dir / "result.json"
                    excel_file = task_output_dir / "predictions.xlsx"

                    if excel_file.exists():
                        # 读取Excel文件中的技能评分数据
                        import pandas as pd
                        try:
                            df = pd.read_excel(excel_file)
                            # 将技能评分转换为字典格式
                            skills_scores = {}
                            for _, row in df.iterrows():
                                skill_tag = row['Class']
                                score = row['Value']
                                skills_scores[skill_tag] = score
                            
                            # 构建返回结果 - 符合前端期望的格式
                            result_data = {
                                'scores_data': skills_scores,  # 前端期望的字段名
                                'task_id': task_id,
                                'voice_type': voice_type,
                                'excel_file': str(excel_file)
                            }
                            
                            logger.info(f"推理完成: 任务={task_id}, 技能评分: {skills_scores}")
                            return result_data
                            
                        except Exception as e:
                            logger.error(f"读取Excel文件失败: {e}")
                            return None
                    else:
                        logger.error(f"未找到Excel结果文件: {excel_file}")
                        return None
                else:
                    logger.error(f"推理失败: 任务={task_id}, 返回码={process.returncode}")
                    return None
                    
            except asyncio.TimeoutError:
                logger.error(f"推理超时: 任务={task_id}")
                process.kill()
                await process.wait()
                return None
                
        except Exception as e:
            logger.error(f"推理异常: 任务={task_id}, 错误={e}")
            return None
    
    
    async def _execute_script(self, input_file: str, task_id: str) -> Optional[float]:
        """
        执行评分脚本
        
        Args:
            input_file: 输入文件路径
            task_id: 任务ID
            
        Returns:
            Optional[float]: 评分结果
        """
        try:
            # 构建命令
            if self.script_path.endswith('.py'):
                # Python脚本
                cmd = ['python', self.script_path, input_file]
            else:
                # 可执行文件
                cmd = [self.script_path, input_file]
            
            logger.debug(f"执行命令: {' '.join(cmd)}")
            
            # 异步执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.output_dir)
            )

            try:
                # 实时输出子进程日志
                stdout_task = asyncio.create_task(self._log_stream(process.stdout, is_error=False))
                stderr_task = asyncio.create_task(self._log_stream(process.stderr, is_error=True))

                # 等待执行完成，设置超时
                await asyncio.wait_for(process.wait(), timeout=self.timeout)
                await stdout_task
                await stderr_task
                
                if process.returncode == 0:
                    # 解析输出结果（此处保留解析逻辑，但由于已实时输出，若需要基于完整stdout可在脚本内写入文件）
                    result = None
                    
                    if result is not None:
                        logger.info(f"模型评分完成: 任务={task_id}, 分数={result}")
                        return result
                    else:
                        logger.error(f"解析评分结果失败: 任务={task_id}，请检查子进程日志输出")
                        return None
                else:
                    logger.error(f"脚本执行失败: 任务={task_id}, 返回码={process.returncode}")
                    return None
                    
            except asyncio.TimeoutError:
                logger.error(f"脚本执行超时: 任务={task_id}, 超时时间={self.timeout}秒")
                process.kill()
                await process.wait()
                return None
                
        except Exception as e:
            logger.error(f"执行脚本异常: 任务={task_id}, 错误={e}")
            return None
    
    def _parse_output(self, output: str, task_id: str) -> Optional[float]:
        """
        解析脚本输出结果
        
        Args:
            output: 脚本输出
            task_id: 任务ID
            
        Returns:
            Optional[float]: 解析出的评分
        """
        try:
            # 尝试直接解析为浮点数
            score = float(output)
            
            # 验证分数范围（假设0-100）
            if 0 <= score <= 100:
                return score
            else:
                logger.warning(f"评分超出范围: 任务={task_id}, 分数={score}")
                return max(0, min(100, score))  # 限制在0-100范围内
                
        except ValueError:
            # 尝试解析JSON格式
            try:
                data = json.loads(output)
                if isinstance(data, dict) and 'score' in data:
                    score = float(data['score'])
                    if 0 <= score <= 100:
                        return score
                    else:
                        return max(0, min(100, score))
                else:
                    logger.error(f"JSON输出格式错误: 任务={task_id}, 数据={data}")
                    return None
            except json.JSONDecodeError:
                logger.error(f"无法解析输出: 任务={task_id}, 输出={output}")
                return None
    
    def get_runner_stats(self) -> Dict[str, Any]:
        """
        获取执行器统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'script_path': self.script_path,
            'output_dir': str(self.output_dir),
            'timeout': self.timeout,
            'script_exists': os.path.exists(self.script_path) if isinstance(self.script_path, str) else False
        }
