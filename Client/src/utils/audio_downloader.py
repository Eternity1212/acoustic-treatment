#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频文件下载模块
"""

import requests
from pathlib import Path
from typing import Optional
from config.config import SERVER_ROOT
from utils.logger import get_logger

logger = get_logger()

class AudioDownloader:
    """音频文件下载器"""
    
    def __init__(self, session, output_dir: str = "outputs"):
        self.session = session
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp_audio"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def download_audio_file(self, audio_path: str, task_id: str) -> Optional[Path]:
        """
        从服务器下载音频文件到本地
        
        Args:
            audio_path: 音频文件在服务器上的相对路径
            task_id: 任务ID，用于生成唯一的本地文件名
            
        Returns:
            Optional[Path]: 下载成功返回本地文件路径，失败返回None
        """
        try:
            # 构建下载URL - 批量拆分文件在 /uploads/client/split/ 路径下
            if audio_path.startswith('http'):
                download_url = audio_path
            elif audio_path.startswith('/'):
                download_url = f"{SERVER_ROOT}{audio_path}"
            else:
                # 批量拆分文件的相对路径，添加uploads前缀
                if audio_path.startswith('client/split/'):
                    download_url = f"{SERVER_ROOT}/uploads/{audio_path}"
                else:
                    # 其他音频文件默认在datasets/audio目录
                    download_url = f"{SERVER_ROOT}/{audio_path}"
            
            # 生成本地文件名
            audio_filename = Path(audio_path).name
            local_path = self.temp_dir / f"task_{task_id}_{audio_filename}"
            
            logger.info(f"📥 开始下载音频文件: {download_url}")
            logger.info(f"📁 保存到: {local_path}")
            
            # 使用requests进行同步下载
            try:
                response = self.session.get(download_url, timeout=30)
                logger.info(f"📊 HTTP响应状态: {response.status_code}")
            except Exception as req_error:
                logger.error(f"❌ 网络请求失败: {req_error}")
                return None
            
            if response and response.status_code == 200:
                # 保存文件内容
                with open(local_path, 'wb') as f:
                    # 分块写入文件
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.info(f"✅ 音频文件下载完成: {local_path}")
                logger.info(f"📊 文件大小: {local_path.stat().st_size} bytes")
                return local_path
            else:
                logger.error(f"❌ 下载失败: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 下载音频文件异常: {e}")
            import traceback
            logger.error(f"❌ 详细错误: {traceback.format_exc()}")
            return None
    
    def cleanup_temp_file(self, file_path: Path):
        """
        清理临时音频文件
        
        Args:
            file_path: 要清理的文件路径
        """
        try:
            if file_path and file_path.exists():
                file_path.unlink()
                logger.debug(f"🗑️ 已清理临时音频文件: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ 清理临时文件失败: {e}")
    
    def cleanup_old_files(self, keep_files: int = 10):
        """
        清理旧的临时音频文件
        
        Args:
            keep_files: 保留最新的文件数量
        """
        try:
            if not self.temp_dir.exists():
                return
            
            # 获取所有音频文件，按修改时间排序
            audio_files = list(self.temp_dir.glob("task_*"))
            audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 删除超出保留数量的文件
            for file_path in audio_files[keep_files:]:
                try:
                    file_path.unlink()
                    logger.debug(f"🗑️ 已清理旧音频文件: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理文件失败: {file_path}, 错误: {e}")
                    
        except Exception as e:
            logger.error(f"❌ 清理旧文件异常: {e}")
