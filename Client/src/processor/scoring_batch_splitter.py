import asyncio
import subprocess
import time
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
import sys

logger = logging.getLogger(__name__)


class ScoringBatchSplitter:
    """在线评分任务大文件拆分处理器"""

    def __init__(self, api_client, audio_downloader):
        """
        初始化拆分处理器

        Args:
            api_client: API客户端实例
            audio_downloader: 音频下载器实例
        """
        self.api_client = api_client
        self.audio_downloader = audio_downloader
        self.running_splits = {}  # 正在处理的拆分任务
        self.completed_splits = []  # 已完成的拆分任务

        self.max_concurrent_splits = 2  # 最大并发拆分任务数
        # 强制指定输出根目录（绝对路径，确保脚本输出到这里）
        self.output_root_dir = r"C:\Users\diva\Desktop\competition-diva-ai-main\Client\outputs\temp_scoring_splits"
        self.temp_download_dir = "outputs/temp_scoring_downloads"
        self.splitting_script_path = r"D:\competition\audio_process_pth.py"  # 拆分脚本绝对路径
        # 可选脚本路径（根据实际情况切换）：D:\competition\audio_process_onnx.py
        self.segment_duration = 10  # 默认10秒分段

        # 声部类型映射
        self.voice_type_names = {
            1: 'sopran',
            2: 'mezzo',
            3: 'falsetto',
            4: 'tenor',
            5: 'baritone',
            6: 'bass'
        }

        self.venv_python = r"D:\xuchengwei\anaconda\envs\competition_py310\python.exe"

        # 确保输出目录和下载目录存在
        Path(self.output_root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_download_dir).mkdir(parents=True, exist_ok=True)

    async def process_scoring_splits(self):
        """处理在线评分任务大文件拆分的主循环"""
        try:
            # 获取待拆分任务
            splits = self.api_client.fetch_scoring_task_splits(limit=self.max_concurrent_splits)

            if not splits:
                return

            logger.info(f"获取到 {len(splits)} 个待拆分任务")

            # 并发处理拆分任务
            tasks = []
            for split_task in splits:
                if len(self.running_splits) < self.max_concurrent_splits:
                    task = asyncio.create_task(self.process_single_scoring_split(split_task))
                    tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"处理拆分任务异常: {e}", exc_info=True)

    async def process_single_scoring_split(self, split_task: Dict[str, Any]):
        """
        处理单个在线评分任务大文件拆分

        Args:
            split_task: 拆分任务信息
        """
        split_id = split_task.get('split_id')
        if not split_id:
            logger.error("拆分任务缺少split_id")
            return False

        try:
            logger.info(f"开始处理拆分任务: {split_id}")

            # 更新状态为processing
            self.api_client.update_scoring_split_status(split_id, 'processing')

            # 记录开始时间
            start_time = time.time()
            self.running_splits[split_id] = {
                'task': split_task,
                'start_time': start_time,
                'status': 'processing'
            }

            # 1. 下载大文件（添加延迟释放文件句柄）
            local_file = await self.download_large_file(split_task)
            if not local_file or not Path(local_file).exists():
                logger.error(f"拆分任务 {split_id} 大文件下载失败")
                self.api_client.update_scoring_split_status(split_id, 'failed')
                return False

            # 2. 执行拆分脚本：每个任务独立输出目录（split_xxx）
            output_dir = Path(self.output_root_dir) / f"split_{split_id}"
            logger.info(f"🎵 即将执行音频拆分: 输入文件={local_file}, 输出目录={output_dir}")
            split_files = await self.execute_scoring_split(local_file, output_dir, split_task)

            if not split_files:
                logger.error(f"拆分任务 {split_id} 拆分失败")
                self.api_client.update_scoring_split_status(split_id, 'failed')
                return False

            # 3. 逐个创建在线评分任务
            success_count = 0
            for split_file in split_files:
                success = self.api_client.create_online_scoring_task(
                    audio_file=str(split_file),
                    singer=split_task['singer_name'],
                    song=split_task['song_name'],
                    voice_type=split_task['voice_type'],
                    uploader_id=split_task['uploader_id']
                )
                if success:
                    success_count += 1

            logger.info(f"拆分任务 {split_id} 成功创建 {success_count}/{len(split_files)} 个在线评分任务")

            # 4. 更新拆分任务状态为completed
            self.api_client.update_scoring_split_status(split_id, 'completed')

            # 5. 记录完成信息
            processing_time = time.time() - start_time
            self.completed_splits.append({
                'split_id': split_id,
                'split_count': len(split_files),
                'success_count': success_count,
                'processing_time': processing_time,
                'completed_at': time.time()
            })

            # 6. 清理临时文件
            await self.cleanup_temp_files(local_file, output_dir)

            logger.info(f"拆分任务 {split_id} 处理完成，耗时 {processing_time:.2f} 秒")
            return True

        except Exception as e:
            logger.error(f"处理拆分任务 {split_id} 异常: {e}", exc_info=True)
            self.api_client.update_scoring_split_status(split_id, 'failed')
            return False
        finally:
            # 清理运行中任务记录
            if split_id in self.running_splits:
                del self.running_splits[split_id]

    async def download_large_file(self, split_task: Dict[str, Any]) -> str:
        """
        下载大文件（添加延迟释放文件句柄）

        Args:
            split_task: 拆分任务信息

        Returns:
            str: 本地文件路径，失败返回None
        """
        try:
            split_id = split_task['split_id']
            file_path = split_task['large_file_path']

            logger.info(f"下载大文件: split_id={split_id}, file_path={file_path}")

            # 使用现有的音频下载器
            local_file = self.audio_downloader.download_audio_file(
                file_path,
                f"scoring_split_{split_id}"
            )

            if local_file and Path(local_file).exists():
                logger.info(f"大文件下载成功: {local_file}（大小：{Path(local_file).stat().st_size / 1024 / 1024:.2f}MB）")
                # 添加1秒延迟，确保文件句柄释放
                await asyncio.sleep(1)
                return local_file
            else:
                logger.error(f"大文件下载失败: {file_path}")
                return None

        except Exception as e:
            logger.error(f"下载大文件异常: {e}", exc_info=True)
            return None

    async def execute_scoring_split(self, input_file: str, output_dir: Path, split_task: Dict[str, Any]) -> List[Path]:
        """
        执行在线评分任务音频拆分（强制输出到指定目录）

        Args:
            input_file: 输入音频文件路径
            output_dir: 输出目录（split_xxx）
            split_task: 拆分任务信息

        Returns:
            List[Path]: 拆分后的文件列表
        """
        try:
            # 确保输出目录存在（强制创建）
            output_dir.mkdir(parents=True, exist_ok=True)
            # 强制转换为绝对路径，避免脚本解析错误
            output_dir_abs = str(output_dir.resolve())
            input_file_abs = str(Path(input_file).resolve())

            # 获取声部名称
            voice_type = split_task['voice_type']
            voice_part = self.voice_type_names.get(voice_type, 'sopran')

            # 构建命令：--output_dir 传递强制指定的绝对路径
            cmd = [
                self.venv_python,  # 虚拟环境Python
                # sys.executable,
                self.splitting_script_path,
                '--input', input_file_abs,  # 输入文件绝对路径
                '--singer', split_task['singer_name'],
                '--song', split_task['song_name'],
                '--part', voice_part,
                '--output_dir', output_dir_abs,  # 输出目录绝对路径（强制指定）
            ]

            # 环境变量：继承主程序所有变量，确保虚拟环境生效
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            # 详细调试输出
            print("=" * 80)
            print("🎵 音频拆分脚本执行详情（强制指定输出目录）")
            print("=" * 80)
            print(f"工作目录: D:/competition")
            print(f"Python解释器: {self.venv_python}")
            print(f"拆分脚本路径: {self.splitting_script_path}")
            print(f"输入文件（绝对路径）: {input_file_abs}")
            print(f"输出目录（绝对路径）: {output_dir_abs}")  # 确认输出目录正确
            print(f"完整命令: {' '.join(cmd)}")
            print("=" * 80)

            # 日志记录
            logger.info("=" * 80)
            logger.info("🎵 音频拆分脚本执行详情（强制指定输出目录）")
            logger.info("=" * 80)
            logger.info(f"📁 工作目录: D:/competition")
            logger.info(f"🐍 Python解释器: {self.venv_python}")
            logger.info(f"📜 拆分脚本路径: {self.splitting_script_path}")
            logger.info(f"📥 输入文件（绝对路径）: {input_file_abs}")
            logger.info(f"📤 输出目录（绝对路径）: {output_dir_abs}")  # 日志中确认路径
            logger.info(f"🔧 完整命令: {' '.join(cmd)}")
            logger.info("=" * 80)

            # 执行脚本：不捕获stdout（避免死锁），捕获stderr（报错排查）
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=None,  # 直接打印到控制台
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=r"D:/competition"  # 保留工作目录（找到UVR依赖）
            )

            logger.info(f"⌛ 拆分脚本执行中（预计10秒内完成，输出到：{output_dir_abs}）...")

            # 等待执行完成（超时30秒）
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
                exit_code = process.returncode
            except asyncio.TimeoutError:
                logger.error("❌ 子进程执行超时（30秒），强制终止！")
                process.kill()
                stdout, stderr = await process.communicate()
                exit_code = process.returncode
                logger.error(f"超时后错误信息: {stderr.decode('utf-8', errors='ignore')}")
                return []

            # 输出执行结果
            logger.info("=" * 80)
            logger.info(f"🔍 脚本执行结果（返回码：{exit_code}）")
            logger.info("=" * 80)

            # 打印错误信息（如果有）
            if stderr:
                try:
                    error_msg = stderr.decode('utf-8').strip()
                except UnicodeDecodeError:
                    error_msg = stderr.decode('gbk').strip()
                if error_msg and '100%' not in error_msg:  # 过滤进度条输出
                    logger.error("⚠️ 脚本错误输出:")
                    for line in error_msg.split('\n'):
                        if line.strip():
                            logger.error(f"   {line}")

            if exit_code == 0:
                logger.info("✅ 拆分脚本执行成功")

                # 强制查找输出目录下的所有WAV文件（递归查找，避免脚本输出到子目录）
                split_files = list(output_dir.glob('**/*.wav'))  # 递归查找所有子目录的WAV
                if split_files:
                    logger.info(f"📊 拆分完成！找到 {len(split_files)} 个10秒片段")
                    logger.info(f"📁 片段路径示例: {[str(f) for f in split_files[:2]]}...")
                    return split_files
                else:
                    logger.error("❌ 脚本执行成功，但未找到拆分文件！")
                    logger.error(f"📂 输出目录内容（含子目录）: {list(output_dir.glob('*'))}")
                    # 打印D:/competition目录下的split_xxx文件夹（排查脚本是否输出到别处）
                    competition_split_dirs = list(Path(r"D:/competition").glob("split_*"))
                    if competition_split_dirs:
                        logger.error(f"⚠️ 发现D:/competition目录下有拆分目录: {competition_split_dirs}")
                    return []
            else:
                logger.error(f"❌ 拆分脚本执行失败（返回码：{exit_code}）")
                return []

        except Exception as e:
            logger.error(f"执行拆分异常: {e}", exc_info=True)
            return []

    async def cleanup_temp_files(self, local_file: str, output_dir: Path):
        """
        清理临时文件

        Args:
            local_file: 下载的临时文件
            output_dir: 拆分输出目录
        """
        try:
            # 清理下载的大文件
            if local_file and Path(local_file).exists():
                self.audio_downloader.cleanup_old_files(keep_files=50)
                Path(local_file).unlink(missing_ok=True)  # 强制删除
                logger.info(f"清理临时下载文件: {local_file}")

            # 可选：清理拆分输出目录（根据需求决定是否保留）
            # if output_dir.exists():
            #     import shutil
            #     shutil.rmtree(output_dir)
            #     logger.info(f"清理拆分输出目录: {output_dir}")

        except Exception as e:
            logger.error(f"清理临时文件异常: {e}", exc_info=True)

    def get_status_summary(self) -> Dict[str, Any]:
        """
        获取拆分处理器状态摘要

        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            'running_splits': len(self.running_splits),
            'completed_splits': len(self.completed_splits),
            'max_concurrent_splits': self.max_concurrent_splits,
            'running_split_ids': list(self.running_splits.keys())
        }