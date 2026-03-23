"""
日志工具模块
"""

import logging
import sys
import os

# 添加src目录到Python路径
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_dir)

# 直接导入常量文件
constant_file = os.path.join(src_dir, 'config', 'constant.py')
import importlib.util
spec = importlib.util.spec_from_file_location("constant", constant_file)
constant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constant)
LOG_LEVELS = constant.LOG_LEVELS
from datetime import datetime
from pathlib import Path

def get_logger(name: str = "client", level: str = "INFO") -> logging.Logger:
    """
    获取配置好的日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(getattr(logging, level.upper()))
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    try:
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"无法创建日志文件: {e}")
    
    return logger
