'''
客户端启动程序
集成自动更新检查功能
'''

import sys
import os
from pathlib import Path

# 预先设置Python路径，确保可以导入项目包和src下模块
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from config.update import UpdateManager

def check_and_update_on_startup():
    """启动时检查更新"""
    try:
        print("正在检查更新...")
        updater = UpdateManager()
        
        # 检查是否有可用更新
        update_info = updater.check_for_updates()
        
        if update_info:
            print(f"发现新版本: {update_info['version']}")
            print(f"更新内容: {update_info['update_content']}")
            
            # 询问用户是否要更新
            while True:
                choice = input("是否现在更新? (y/n): ").lower().strip()
                if choice in ['y', 'yes', '是']:
                    print("开始更新...")
                    success = updater.perform_update()
                    if success:
                        print("更新完成，请重新启动程序")
                        return False  # 需要重启
                    else:
                        print("更新失败，继续使用当前版本")
                        break
                elif choice in ['n', 'no', '否']:
                    print("跳过更新，继续启动")
                    break
                else:
                    print("请输入 y 或 n")
        else:
            print("当前已是最新版本")
            
        return True  # 可以继续启动
        
    except Exception as e:
        print(f"检查更新时发生错误: {e}")
        print("跳过更新检查，继续启动")
        return True

def main():
    """主启动函数"""
    print("客户端启动中...")
    
    # 启动时检查更新
    can_continue = check_and_update_on_startup()
    
    if not can_continue:
        print("程序需要重启以应用更新")
        return
    
    # 这里添加你的主程序逻辑
    print("启动完成，开始运行主程序...")
    
    # 导入并运行主程序模块
    try:
        # 检查src目录是否存在
        src_dir = Path(__file__).parent / "src"
        if src_dir.exists() and any(src_dir.iterdir()):
            # 添加项目根目录和src目录到Python路径
            project_root = Path(__file__).parent
            sys.path.insert(0, str(project_root))
            sys.path.insert(0, str(src_dir))
            
            # 导入主程序模块
            import main
            import asyncio
            
            print("启动模型评分客户端...")
            
            # 运行主程序
            asyncio.run(main.main())
            
        else:
            print("src目录为空或不存在，请先更新程序")
            
    except ImportError as e:
        print(f"导入主程序模块失败: {e}")
        print("请检查src目录中是否有正确的程序文件")
    except Exception as e:
        print(f"运行主程序时发生错误: {e}")

if __name__ == "__main__":
    main()