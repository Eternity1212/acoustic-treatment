"""
文件操作工具模块
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

def ensure_dir(path: str) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path: 目录路径对象
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def cleanup_temp_files(temp_dir: str = None) -> bool:
    """
    清理临时文件
    
    Args:
        temp_dir: 临时目录路径，如果为None则使用系统临时目录
        
    Returns:
        bool: 清理是否成功
    """
    try:
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        temp_path = Path(temp_dir)
        if temp_path.exists():
            # 清理以client_开头的临时文件
            for file in temp_path.glob("client_*"):
                try:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(file)
                except Exception as e:
                    print(f"清理临时文件失败 {file}: {e}")
        
        return True
    except Exception as e:
        print(f"清理临时文件异常: {e}")
        return False

def get_file_size(file_path: str) -> int:
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
        
    Returns:
        int: 文件大小(字节)
    """
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0

def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小显示
    
    Args:
        size_bytes: 文件大小(字节)
        
    Returns:
        str: 格式化后的大小字符串
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def create_temp_file(suffix: str = ".tmp", prefix: str = "client_") -> str:
    """
    创建临时文件
    
    Args:
        suffix: 文件后缀
        prefix: 文件前缀
        
    Returns:
        str: 临时文件路径
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)  # 关闭文件描述符
    return temp_path

def safe_remove_file(file_path: str) -> bool:
    """
    安全删除文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 删除是否成功
    """
    try:
        file_obj = Path(file_path)
        if file_obj.exists():
            file_obj.unlink()
        return True
    except Exception as e:
        print(f"删除文件失败 {file_path}: {e}")
        return False
