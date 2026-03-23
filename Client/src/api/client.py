"""
API客户端模块
处理与后端API的所有HTTP通信
"""

import requests
import os
from typing import Dict, List, Any
from pathlib import Path
import time

# 使用importlib直接导入配置模块
import importlib.util

# 导入config.config模块
config_spec = importlib.util.spec_from_file_location(
    "config", 
    Path(__file__).parent.parent.parent / "config" / "config.py"
)
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)

# 从配置模块获取变量
API_KEY = config_module.API_KEY
MODEL_SCORING_URL = config_module.MODEL_SCORING_URL

class APIClient:
    """API客户端类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = API_KEY
        self.session = requests.Session()
        self.base_headers = {
            'Authorization': f'ApiKey {self.api_key}',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Cache-Control': 'max-age=0',
            'Priority': 'u=0, i',
            'Sec-Ch-Ua': '"Not;A=Brand";v="99", "Microsoft Edge";v="139", "Chromium";v="139"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0'
        }
        
        # 配置requests会话
        self.session.headers.update(self.base_headers)
        self.session.verify = False  # 禁用SSL验证
        
        # 配置连接适配器，针对Windows网络问题优化
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        import urllib3
        
        # 禁用连接池和持久连接
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 配置重试策略，针对连接中断问题
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PATCH"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=1,  # 减少连接池大小
            pool_maxsize=1,      # 减少最大连接数
            pool_block=False
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置会话级别的配置
        self.session.trust_env = False  # 不使用环境变量的代理设置
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.session:
            self.session.close()
    
    def _make_request(self, method: str, url: str, **kwargs):
        """发起HTTP请求的通用方法"""
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败：方法={method}，URL={url}，异常类型={type(e).__name__}，详情={e}")
            return None
        except Exception as e:
            print(f"请求异常: {e}")
            return None
    
    def fetch_pending_tasks(self, limit: int = 10) -> List[Dict]:
        """
        获取待评分任务列表
        
        Args:
            limit: 获取任务数量限制
            
        Returns:
            List[Dict]: 待评分任务列表
        """
        try:
            url = MODEL_SCORING_URL['pending_tasks']
            params = {'limit': limit}

            print(url)
            print(f"获取待评分任务")
            # print(f"请求URL: {url}")
            # print(f"请求参数: {params}")
            # print(f"请求头: {self.base_headers}")
            
            response = self.session.get(url, params=params)
            print(f"响应状态码: {response.status_code}")
            # print(f"响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                # print(f"响应文本内容: {response.text}")  # 看是否为空或非JSON
                # print(f"响应字节内容: {response.content}")  # 排除不可见字符
                # print(f"响应Content-Type: {response.headers.get('Content-Type')}")  # 看是否是application/json
                # print("当前session的Cookie：", self.session.cookies.get_dict())
                data = response.json()
                print(f"成功获取 {len(data.get('tasks', []))} 个任务")
                tasks = data.get('data', [])
                
                if tasks:
                    print(f"获取到 {len(tasks)} 个待评分任务")
                    for i, task in enumerate(tasks):
                        print(f"任务 {i+1}: ID={task.get('result_id')}, 文件={task.get('original_filename')}")
                else:
                    print("暂无待评分任务")
                
                return tasks
            else:
                print(f"获取任务失败: HTTP {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"获取任务异常: {e}")
            return []
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            
        Returns:
            bool: 更新是否成功
        """
        try:
            url = f"{MODEL_SCORING_URL['base']}/task/{task_id}/status"
            data = {'status': status}
            
            print(f"更新任务状态")
            # print(f"请求URL: {url}")
            # print(f"请求数据: {data}")
            # print(f"请求头: {self.base_headers}")
            
            response = self.session.patch(url, json=data)
            # print(f"响应状态码: {response.status_code}")
            # print(f"响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                print("任务状态更新成功")
                return True
            else:
                print(f"更新状态失败: HTTP {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"更新状态异常: {e}")
            return False
    
    def submit_result(self, task_id: str, model_result: dict, processing_time: float = None) -> bool:
        """
        提交模型评分结果
        
        Args:
            task_id: 任务ID
            model_result: 模型评分结果
            processing_time: 处理时间(秒)
            
        Returns:
            bool: 提交是否成功
        """
        try:
            url = f"{MODEL_SCORING_URL['base']}/task/{task_id}/complete"
            data = {
                'modelResult': model_result
            }
            
            if processing_time is not None:
                data['processingTime'] = processing_time
            
            print(f"提交处理结果")

            response = self.session.post(url, json=data)

            if response.status_code == 200:
                print("处理结果提交成功")
                return True
            else:
                print(f"提交失败: HTTP {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"提交结果异常: {e}")
            return False
    
    def fetch_scoring_task_splits(self, limit: int = 3) -> List[Dict]:
        """
        获取待拆分的在线评分任务大文件
        
        Args:
            limit: 获取任务数量限制
            
        Returns:
            List[Dict]: 拆分任务列表
        """
        try:
            url = f"{MODEL_SCORING_URL['base'].replace('/model-scoring', '')}/client/scoring-task-splits/pending"
            params = {'limit': limit}
            
            response = self._make_request('GET', url, params=params)
            if response and response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('data', [])
            return []
        except Exception as e:
            print(f"获取拆分任务失败: {e}")
            return []
    
    def update_scoring_split_status(self, split_id: str, status: str) -> bool:
        """
        更新在线评分任务拆分状态
        
        Args:
            split_id: 拆分任务ID
            status: 状态 (pending/processing/completed/failed)
            
        Returns:
            bool: 更新是否成功
        """
        try:
            url = f"{MODEL_SCORING_URL['base'].replace('/model-scoring', '')}/client/scoring-task-splits/{split_id}/status"
            data = {'status': status}
            
            print(f"更新拆分任务状态: split_id={split_id}, status={status}")
            
            response = self.session.patch(url, json=data)
            
            if response.status_code == 200:
                print("拆分任务状态更新成功")
                return True
            else:
                print(f"更新失败: HTTP {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"更新拆分任务状态异常: {e}")
            return False
    
    def upload_audio_file(self, audio_file_path: str, uploader_id: int) -> str:
        """
        上传音频文件到服务器
        
        Args:
            audio_file_path: 本地音频文件路径
            uploader_id: 上传者用户ID
            
        Returns:
            str: 服务器上的文件路径，失败返回None
        """
        try:
            if not os.path.exists(audio_file_path):
                print(f"文件不存在: {audio_file_path}")
                return None
                
            url = f"{MODEL_SCORING_URL['base'].replace('/model-scoring', '')}/scoring/client-upload"
            
            with open(audio_file_path, 'rb') as f:
                # 确保文件名使用UTF-8编码
                filename = os.path.basename(audio_file_path)
                files = {'audioFiles': (filename.encode('utf-8').decode('utf-8'), f, 'audio/wav')}
                data = {'uploaderId': uploader_id}
                
                print(f"上传音频文件: {os.path.basename(audio_file_path)}")
                response = self.session.post(url, files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                if result.get('success') and result.get('data'):
                    data = result['data']
                    # 检查是否有processedFiles数组
                    if 'processedFiles' in data and len(data['processedFiles']) > 0:
                        server_path = data['processedFiles'][0]['file_path']
                        print(f"文件上传成功: {server_path}")
                        return server_path
                    else:
                        print(f"上传失败: 响应中没有找到处理后的文件信息")
                        return None
                else:
                    print(f"上传失败: {result.get('message', '未知错误')}")
                    return None
            else:
                print(f"上传失败: HTTP {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"上传音频文件异常: {e}")
            return None

    def create_online_scoring_task(self, audio_file: str, singer: str, song: str, voice_type: int, uploader_id: int = 1) -> bool:
        """
        创建在线评分任务（先上传文件再创建任务）
        
        Args:
            audio_file: 本地音频文件路径
            singer: 歌手名
            song: 歌曲名
            voice_type: 声部类型
            uploader_id: 上传者用户ID
            
        Returns:
            bool: 创建是否成功
        """
        try:
            # 1. 先上传音频文件到服务器
            server_file_path = self.upload_audio_file(audio_file, uploader_id)
            if not server_file_path:
                print("文件上传失败，无法创建评分任务")
                return False
            
            # 2. 使用服务器文件路径创建评分任务
            url = f"{MODEL_SCORING_URL['base'].replace('/model-scoring', '')}/scoring/create-task"
            data = {
                'file_path': server_file_path,
                'original_filename': os.path.basename(audio_file),
                'singer_name': singer,
                'song_name': song,
                'voice_type': voice_type,
                'uploader_id': uploader_id
            }
            
            print(f"创建在线评分任务: {singer}-{song}")
            print(f"发送数据: {data}")
            
            # 确保请求头包含正确的编码信息
            headers = {
                'Content-Type': 'application/json; charset=utf-8',
                'X-API-Key': API_KEY
            }
            
            response = self.session.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"在线评分任务创建成功，任务ID: {result.get('data', {}).get('taskId')}")
                    return True
                else:
                    print(f"创建失败: {result.get('message', '未知错误')}")
                    return False
            else:
                print(f"创建失败: HTTP {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"创建在线评分任务异常: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            bool: 服务是否健康
        """
        try:
            url = MODEL_SCORING_URL['health_check']
            print(f"开始健康检查")
            print(f"请求URL: {url}")
            print(f"请求头: {dict(self.base_headers)}")
            
            response = self.session.get(
                url, 
                timeout=(10, 30),
                allow_redirects=True,
                stream=False
            )
            
            print(f"响应状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"响应数据: {data}")
                    return data.get('success', False) or data.get('status') == 'healthy' or data.get('status') == 'OK'
                except:
                    print(f"响应内容: {response.text[:200]}")
                    return True
            
            return False
            
        except requests.exceptions.Timeout as e:
            print(f"健康检查超时: {e}")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"健康检查连接失败: {e}")
            print("可能的原因: 网络连接问题、防火墙阻止、或服务器不可达")
            return False
        except requests.exceptions.SSLError as e:
            print(f"SSL错误: {e}")
            return False
        except Exception as e:
            print(f"健康检查异常: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False
    
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        获取连接状态信息
        
        Returns:
            Dict: 连接状态信息
        """
        return {
            'connected': self.session is not None and not self.session.closed,
            'api_key_configured': bool(self.api_key),
            'base_url': MODEL_SCORING_URL['pending_tasks'].split('/model-scoring')[0]
        }