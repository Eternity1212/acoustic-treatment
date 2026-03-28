'''
客户端更新模块
通过API获取更新信息并下载更新文件覆盖src目录
'''

import os
import sys
import json
import zipfile
import shutil
import tempfile
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple

# 导入配置
from .config import UPDATE_FILE_URL, API_KEY, VERSION, STATIC_URL

class UpdateManager:
    """更新管理器"""
    
    def __init__(self):
        self.current_version = VERSION
        self.api_key = API_KEY
        self.update_url = UPDATE_FILE_URL
        self.static_url = STATIC_URL
        
        # 获取项目根目录和src目录路径
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.backup_dir = self.project_root / "backup"
        
        # 确保目录存在
        self.src_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
    
    def check_for_updates(self) -> Optional[Dict]:
        """
        检查是否有可用更新
        
        Returns:
            Dict: 更新信息，如果没有更新则返回None
        """
        try:
            print(f"🔍 检查更新中... 当前版本: {self.current_version}")
            
            # 请求头
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 发送请求获取更新信息
            response = requests.get(
                self.update_url,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                update_info = response.json()
                
                # 检查返回的数据结构
                if not all(key in update_info for key in ['version', 'update_content', 'update_file']):
                    print("服务器返回的更新信息格式不正确")
                    return None
                
                server_version = update_info['version']
                
                # 版本比较
                if self._compare_versions(server_version, self.current_version) > 0:
                    print(f"发现新版本: {server_version}")
                    print(f"更新内容: {update_info['update_content']}")
                    return update_info
                else:
                    print(f"当前已是最新版本: {self.current_version}")
                    return None
                    
            elif response.status_code == 404:
                print("更新服务不可用")
                return None
            else:
                print(f"检查更新失败: HTTP {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"解析服务器响应失败: {e}")
            return None
        except Exception as e:
            print(f"检查更新时发生未知错误: {e}")
            return None
    
    def download_update(self, update_info: Dict) -> Optional[str]:
        """
        下载更新文件
        
        Args:
            update_info: 更新信息
            
        Returns:
            str: 下载的文件路径，失败返回None
        """
        try:
            update_file_path = update_info['update_file']
            
            # 构建完整的下载URL
            if update_file_path.startswith('http'):
                download_url = update_file_path
            else:
                download_url = f"{self.static_url.rstrip('/')}/{update_file_path.lstrip('/')}"
            
            print(f"开始下载更新文件: {download_url}")
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_file_path = temp_file.name
            temp_file.close()
            
            # 下载文件
            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # 显示下载进度
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r下载进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n更新文件下载完成: {temp_file_path}")
            return temp_file_path
            
        except requests.exceptions.RequestException as e:
            print(f"\n下载更新文件失败: {e}")
            return None
        except Exception as e:
            print(f"\n下载过程中发生未知错误: {e}")
            return None
    
    def backup_current_version(self) -> bool:
        """
        备份当前版本的src目录
        
        Returns:
            bool: 备份是否成功
        """
        try:
            if not self.src_dir.exists() or not any(self.src_dir.iterdir()):
                print("src目录为空，跳过备份")
                return True
            
            # 创建备份文件名
            backup_name = f"backup_v{self.current_version}_{int(os.path.getmtime(self.src_dir))}"
            backup_path = self.backup_dir / backup_name
            
            print(f"备份当前版本到: {backup_path}")
            
            # 如果备份目录已存在，先删除
            if backup_path.exists():
                shutil.rmtree(backup_path)
            
            # 复制src目录到备份位置
            shutil.copytree(self.src_dir, backup_path)
            
            print("备份完成")
            return True
            
        except Exception as e:
            print(f"备份失败: {e}")
            return False
    
    def extract_and_replace(self, zip_file_path: str) -> bool:
        """
        解压更新文件并替换src目录
        
        Args:
            zip_file_path: 更新文件的路径
            
        Returns:
            bool: 更新是否成功
        """
        try:
            print("解压更新文件...")
            
            # 创建临时解压目录
            with tempfile.TemporaryDirectory() as temp_extract_dir:
                
                # 解压文件
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                # 查找解压后的内容
                extract_path = Path(temp_extract_dir)
                
                # 检查是否有src目录或直接是文件
                src_in_zip = extract_path / "src"
                if src_in_zip.exists():
                    source_path = src_in_zip
                else:
                    # 如果没有src目录，假设解压的内容就是要放到src的文件
                    source_path = extract_path
                
                # 清空现有的src目录
                if self.src_dir.exists():
                    print("清空现有src目录...")
                    shutil.rmtree(self.src_dir)
                
                # 创建新的src目录
                self.src_dir.mkdir(exist_ok=True)
                
                # 复制新文件到src目录
                print("复制新文件到src目录...")
                if source_path.is_dir():
                    for item in source_path.iterdir():
                        if item.is_dir():
                            shutil.copytree(item, self.src_dir / item.name)
                        else:
                            shutil.copy2(item, self.src_dir / item.name)
                else:
                    # 如果source_path是文件，直接复制
                    shutil.copy2(source_path, self.src_dir)
                
                print("文件替换完成")
                return True
                
        except zipfile.BadZipFile:
            print("更新文件不是有效的ZIP文件")
            return False
        except Exception as e:
            print(f"解压和替换过程中发生错误: {e}")
            return False
        finally:
            # 清理临时文件
            try:
                if os.path.exists(zip_file_path):
                    os.unlink(zip_file_path)
            except:
                pass
    
    def update_version_info(self, new_version: str) -> bool:
        """
        更新版本信息到config.py
        
        Args:
            new_version: 新版本号
            
        Returns:
            bool: 更新是否成功
        """
        try:
            config_file = self.project_root / "config" / "config.py"
            
            if not config_file.exists():
                print("config.py文件不存在")
                return False
            
            # 读取配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换版本号
            import re
            new_content = re.sub(
                r"VERSION\s*=\s*['\"].*?['\"]",
                f"VERSION = '{new_version}'",
                content
            )
            
            # 写回文件
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"版本信息已更新为: {new_version}")
            return True
            
        except Exception as e:
            print(f"更新版本信息失败: {e}")
            return False
    
    def perform_update(self) -> bool:
        """
        执行完整的更新流程
        
        Returns:
            bool: 更新是否成功
        """
        print(" 开始检查和执行更新...")
        
        # 1. 检查更新
        update_info = self.check_for_updates()
        if not update_info:
            return False
        
        # 2. 备份当前版本
        if not self.backup_current_version():
            print("备份失败，取消更新")
            return False
        
        # 3. 下载更新文件
        zip_file_path = self.download_update(update_info)
        if not zip_file_path:
            print("下载失败，取消更新")
            return False
        
        # 4. 解压并替换文件
        if not self.extract_and_replace(zip_file_path):
            print("文件替换失败，取消更新")
            return False
        
        # 5. 更新版本信息
        new_version = update_info['version']
        if not self.update_version_info(new_version):
            print("版本信息更新失败，但文件已更新")
        
        print(f"更新完成！版本: {self.current_version} -> {new_version}")
        print("建议重启应用程序以确保更新生效")
        
        return True
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        比较两个版本号
        
        Args:
            version1: 版本1
            version2: 版本2
            
        Returns:
            int: 1表示version1更新，-1表示version2更新，0表示相同
        """
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        
        v1_tuple = version_tuple(version1)
        v2_tuple = version_tuple(version2)
        
        if v1_tuple > v2_tuple:
            return 1
        elif v1_tuple < v2_tuple:
            return -1
        else:
            return 0


def check_and_update():
    """检查并执行更新的便捷函数"""
    updater = UpdateManager()
    return updater.perform_update()


def check_updates_only():
    """仅检查更新，不执行更新"""
    updater = UpdateManager()
    return updater.check_for_updates()


if __name__ == "__main__":
    # 命令行使用
    import argparse
    
    parser = argparse.ArgumentParser(description='客户端更新工具')
    parser.add_argument('--check-only', action='store_true', help='仅检查更新，不执行')
    parser.add_argument('--force', action='store_true', help='强制执行更新')
    
    args = parser.parse_args()
    
    updater = UpdateManager()
    
    if args.check_only:
        update_info = updater.check_for_updates()
        if update_info:
            print(f"有可用更新: {update_info['version']}")
            sys.exit(0)
        else:
            print("没有可用更新")
            sys.exit(1)
    else:
        success = updater.perform_update()
        sys.exit(0 if success else 1)