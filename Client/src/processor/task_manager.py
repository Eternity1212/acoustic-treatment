"""
任务管理器模块
"""

import asyncio
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from api.client import APIClient
from utils.logger import get_logger
from utils.audio_downloader import AudioDownloader

# 添加src目录到Python路径
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_dir)

# 直接导入常量文件
constant_file = os.path.join(src_dir, 'config', 'constant.py')
import importlib.util
spec = importlib.util.spec_from_file_location("constant", constant_file)
constant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constant)
TASK_STATUS = constant.TASK_STATUS

logger = get_logger("task_manager")

class TaskManager:
    """任务管理器，负责任务的获取、分发和状态管理"""
    
    def __init__(self, api_client, max_concurrent_tasks: int = 3, output_dir: str = "outputs"):
        self.api_client = api_client
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        self.audio_downloader = AudioDownloader(api_client.session, output_dir)
        
        logger.info(f"任务管理器初始化完成，最大并发任务数: {self.max_concurrent_tasks}")
    
    async def fetch_tasks(self, limit: int = 10) -> List[Dict]:
        """
        获取待处理任务
        
        Args:
            limit: 获取任务数量限制
            
        Returns:
            List[Dict]: 任务列表
        """
        try:
            tasks = self.api_client.fetch_pending_tasks(limit)
            logger.info(f"获取到 {len(tasks)} 个待处理任务")
            return tasks
        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return []
    
    async def process_task(self, task: Dict, model_runner) -> bool:
        """
        处理单个任务
        
        Args:
            task: 任务信息
            model_runner: 模型执行器实例
            
        Returns:
            bool: 处理是否成功
        """
        task_id = task.get('result_id') or task.get('task_id')
        if not task_id:
            logger.error("任务缺少result_id或task_id")
            return False
        
        try:
            logger.info(f"开始处理任务: {task_id}")
            
            # 更新任务状态为处理中
            self.api_client.update_task_status(task_id, 'processing')
            
            # 记录开始时间
            start_time = time.time()
            self.running_tasks[task_id] = {
                'task': task,
                'start_time': start_time,
                'status': 'processing'
            }
            
            # 1. 下载音频文件（调试和生产模式都需要）
            audio_path = task.get('audio_path')
            if not audio_path:
                logger.error(f"任务 {task_id} 缺少音频文件路径")
                return False
            
            logger.info(f"下载音频文件: {audio_path}")
            local_audio_path = self.audio_downloader.download_audio_file(audio_path, task_id)
            if not local_audio_path:
                logger.error(f"任务 {task_id} 音频文件下载失败")
                return False
            
            # 2. 更新任务信息，添加本地音频路径
            task_with_audio = task.copy()
            task_with_audio['audio_file_path'] = str(local_audio_path)
            
            try:
                # 3. 执行模型评分（根据IS_DEBUG决定使用真实或模拟评分）
                result = await model_runner.run_scoring(task_with_audio)
                
                if result is not None:
                    # 计算处理时间
                    processing_time = time.time() - start_time
                    
                    # 提交结果
                    success = self.api_client.submit_result(
                        task_id, 
                        result, 
                        processing_time
                    )
                    
                    if success:
                        logger.info(f"任务 {task_id} 处理成功，评分: {result}")
                        self.completed_tasks.append({
                            'task_id': task_id,
                            'score': result,
                            'processing_time': processing_time,
                            'completed_at': time.time()
                        })
                        
                        # 更新任务状态为完成
                        self.api_client.update_task_status(task_id, 'completed')
                        return True
                    else:
                        logger.error(f"任务 {task_id} 结果提交失败")
                        self.api_client.update_task_status(task_id, 'failed')
                        return False
                else:
                    logger.error(f"任务 {task_id} 模型评分失败")
                    self.api_client.update_task_status(task_id, 'failed')
                    return False
                    
            finally:
                print()
                # 4. 清理临时音频文件
                if local_audio_path:
                    self.audio_downloader.cleanup_old_files(keep_files=50)
                
        except Exception as e:
            logger.error(f"处理任务 {task_id} 异常: {e}")
            self.api_client.update_task_status(task_id, 'failed')
            return False
        finally:
            # 清理运行中任务记录
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def process_tasks_batch(self, tasks: List[Dict], model_runner) -> List[bool]:
        """
        批量处理任务
        
        Args:
            tasks: 任务列表
            model_runner: 模型执行器实例
            
        Returns:
            List[bool]: 每个任务的处理结果
        """
        if not tasks:
            return []
        
        logger.info(f"开始批量处理 {len(tasks)} 个任务")
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await self.process_task(task, model_runner)
        
        # 并发处理任务
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"任务 {i} 处理异常: {result}")
                processed_results.append(False)
            else:
                processed_results.append(result)
        
        success_count = sum(1 for r in processed_results if r is True)
        logger.info(f"批量处理完成: {success_count}/{len(tasks)} 成功")
        
        return processed_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取任务处理统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_processed': len(self.completed_tasks) + len(self.failed_tasks),
            'success_rate': (
                len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks))
                if (len(self.completed_tasks) + len(self.failed_tasks)) > 0 else 0
            )
        }
    
    def cleanup_old_records(self, max_records: int = 1000):
        """
        清理旧的任务记录
        
        Args:
            max_records: 保留的最大记录数
        """
        if len(self.completed_tasks) > max_records:
            self.completed_tasks = self.completed_tasks[-max_records:]
        
        if len(self.failed_tasks) > max_records:
            self.failed_tasks = self.failed_tasks[-max_records:]
        
        logger.debug("清理旧任务记录完成")
