"""
连接管理器模块
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger("connection_manager")

class ConnectionManager:
    """连接管理器，处理网络连接和重试逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.connected = False
        self.retry_count = 0
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """建立连接"""
        try:
            if self.session and not self.session.closed:
                return True
            
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.connected = True
            self.retry_count = 0
            
            logger.info("连接管理器已初始化")
            return True
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.connected = False
        logger.info("连接已断开")
    
    async def make_request_with_retry(self, method: str, url: str, **kwargs) -> Optional[aiohttp.ClientResponse]:
        """
        带重试的请求方法
        
        Args:
            method: HTTP方法
            url: 请求URL
            **kwargs: 其他请求参数
            
        Returns:
            Optional[aiohttp.ClientResponse]: 响应对象或None
        """
        for attempt in range(self.max_retries + 1):
            try:
                if not self.session or self.session.closed:
                    await self.connect()
                
                async with self.session.request(method, url, **kwargs) as response:
                    self.retry_count = 0  # 重置重试计数
                    return response
                    
            except aiohttp.ClientError as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"请求最终失败: {e}")
                    self.connected = False
                    return None
                    
            except Exception as e:
                logger.error(f"请求异常: {e}")
                return None
        
        return None
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.session or self.session.closed:
                return False
            
            # 这里可以添加具体的健康检查逻辑
            # 比如ping一个健康检查端点
            return True
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        return {
            'connected': self.connected,
            'session_active': self.session is not None and not self.session.closed,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
