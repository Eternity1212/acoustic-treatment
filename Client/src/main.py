"""
主程序入口模块
"""

import asyncio
import time
from typing import Dict, Any
import sys
import os
from api.client import APIClient
from processor.task_manager import TaskManager
from processor.model_runner import ModelRunner
from processor.scoring_batch_splitter import ScoringBatchSplitter
from utils.logger import get_logger

# 添加项目根目录到Python路径
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# 添加src目录到Python路径  
src_dir = os.path.dirname(__file__)
sys.path.insert(0, src_dir)

# 导入配置文件
config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
config_file = os.path.join(config_dir, 'config.py')

# 直接导入配置模块
import importlib.util
spec = importlib.util.spec_from_file_location("config", config_file)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

# 导入模块
from config.config import MODEL_SCORING_FETCH_INTERVAL
from utils.file_utils import ensure_dir

logger = get_logger("main")

class ModelScoringClient:
    """模型评分客户端主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_client = None
        self.task_manager = None
        self.model_runner = None
        self.scoring_batch_splitter = None
        self.running = False
        
        # 创建必要目录
        ensure_dir(config.get('output_dir', 'outputs'))
        ensure_dir('outputs/temp_scoring_splits')
        ensure_dir('outputs/temp_scoring_downloads')
        ensure_dir('logs')
        
        logger.info("模型评分客户端初始化完成")
        logger.info(f"输出目录: {config.get('output_dir', 'outputs')}")
        logger.info("模型评分客户端初始化完成")
        logger.info(f"输出目录: {config.get('output_dir', 'outputs')}")
    
    def __enter__(self):
        """上下文管理器入口"""
        logger.info("模型评分客户端初始化完成")
        logger.info(f"输出目录: {self.config.get('output_dir', 'outputs')}")
        
        # 初始化API客户端
        self.api_client = APIClient(self.config)
        self.api_client.__enter__()
        
        # 初始化任务管理器和模型运行器
        self.task_manager = TaskManager(self.api_client, output_dir=self.config.get('output_dir', 'outputs'))
        self.model_runner = ModelRunner(self.config)
        
        # 初始化在线评分任务拆分处理器
        from utils.audio_downloader import AudioDownloader
        audio_downloader = AudioDownloader(self.api_client.session, self.config.get('output_dir', 'outputs'))
        self.scoring_batch_splitter = ScoringBatchSplitter(self.api_client, audio_downloader)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.running = False
        
        if self.api_client:
            self.api_client.__exit__(exc_type, exc_val, exc_tb)
    
    async def _process_task(self, task: Dict[str, Any]):
        """处理单个任务（使用完整的task_manager流程）"""
        return await self.task_manager.process_task(task, self.model_runner)
    
    
    def health_check(self) -> bool:
        """健康检查"""
        logger.info("执行健康检查...")
        
        try:
            # API健康检查
            api_healthy = self.api_client.health_check()
            
            # 配置检查
            config_valid = self._validate_config()
            
            if api_healthy and config_valid:
                logger.info("健康检查通过")
                return True
            else:
                logger.error("健康检查失败")
                return False
                
        except Exception as e:
            logger.error(f"健康检查异常: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """验证配置"""
        try:
            # 检查必需配置
            required_keys = [
                'output_dir',
                'max_concurrent_tasks',
                'fetch_interval'
            ]
            
            for key in required_keys:
                if key not in self.config:
                    logger.error(f"配置缺失: {key}")
                    return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证异常: {e}")
            return False
    
    async def run_once(self) -> int:
        """运行一次任务获取和处理"""
        try:
            logger.info("开始一轮任务处理...")
            
            # 获取待处理任务
            tasks = await self.task_manager.fetch_tasks(
                limit=self.config.get('batch_size', 10)
            )
            
            if not tasks:
                logger.info("暂无待处理任务")
                return 0
            
            # 批量处理任务
            results = await self.task_manager.process_tasks_batch(
                tasks, self.model_runner
            )
            
            # 统计结果
            success_count = sum(1 for r in results if r is True)
            
            logger.info(f"本轮处理完成: 处理 {len(tasks)} 个任务, 成功 {success_count} 个")
            
            return success_count
            
        except Exception as e:
            logger.error(f"运行一轮任务处理异常: {e}")
            return 0
    
    async def run(self):
        """运行客户端"""
        logger.info("开始运行模型评分客户端...")
        
        # 执行健康检查
        if not self.health_check():
            logger.error("健康检查失败，无法启动")
            return
        
        self.running = True
        
        try:
            while self.running:
                # # 处理模型评分任务（现有功能）
                await self._process_model_scoring_tasks()
                
                # 处理在线评分任务大文件拆分（新功能）
                await self._process_scoring_splits()
                
                await asyncio.sleep(MODEL_SCORING_FETCH_INTERVAL)
                    
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止客户端...")
        except Exception as e:
            logger.error(f"客户端运行异常: {e}")
    
    async def _process_model_scoring_tasks(self):
        """处理模型评分任务（现有功能）"""
        try:
            # 获取待处理任务
            tasks = self.api_client.fetch_pending_tasks()
            
            if not tasks:
                logger.info("暂无待评分任务")
                return
            
            logger.info(f"获取到 {len(tasks)} 个待评分任务")
            
            # 处理任务
            for task in tasks:
                if not self.running:
                    break
                
                try:
                    await self._process_task(task)
                except Exception as e:
                    logger.error(f"处理模型评分任务异常: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"处理模型评分任务批次异常: {e}")
    
    async def _process_scoring_splits(self):
        """处理在线评分任务大文件拆分（新功能）"""
        try:
            if self.scoring_batch_splitter:
                await self.scoring_batch_splitter.process_scoring_splits()
        except Exception as e:
            logger.error(f"处理拆分任务异常: {e}")
    
    def stop(self):
        """停止客户端"""
        logger.info("正在停止客户端...")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        status = {
            "running": self.running,
            "config": {
                "output_dir": self.config.get('output_dir'),
                "fetch_interval": self.config.get('fetch_interval'),
                "max_concurrent_tasks": self.config.get('max_concurrent_tasks')
            }
        }
        
        if self.task_manager:
            status["task_stats"] = self.task_manager.get_statistics()
        
        if self.model_runner:
            status["runner_stats"] = self.model_runner.get_runner_stats()
        
        if self.api_client:
            status["connection_status"] = self.api_client.get_connection_status()
        
        return status


async def create_client(config: Dict[str, Any]) -> ModelScoringClient:
    """
    创建客户端实例
    
    Args:
        config: 配置字典
        
    Returns:
        ModelScoringClient: 客户端实例
    """
    return ModelScoringClient(config)


def run_client_once(config: Dict[str, Any]) -> int:
    """
    运行客户端一次
    
    Args:
        config: 配置字典
        
    Returns:
        int: 处理的任务数量
    """
    with ModelScoringClient(config) as client:
        return client.run_once()


def load_config() -> Dict[str, Any]:
    """加载配置"""
    return cfg.DEFAULT_CONFIG.copy()

async def main():
    """主函数"""
    try:
        config = load_config()
        
        with ModelScoringClient(config) as client:
            await client.run()
            
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行异常: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
