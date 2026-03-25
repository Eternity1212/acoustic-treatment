'''
配置文件 存储各个API的地址
'''

# 版本号
VERSION = '1.0.0'

# 是否是国际版本
IS_INTERNATIONAL = False

# 是否本地调试
IS_DEBUG = False

# 接口地址
API_URL = 'http://localhost:3000' if IS_DEBUG else 'https://api.diva-ai.cn/api' if not IS_INTERNATIONAL else 'https://api.diva-ai.cn/api/apiservice/api'
STATIC_URL = 'http://localhost:3000/upload' if IS_DEBUG else 'https://api.diva-ai.cn/upload' if not IS_INTERNATIONAL else 'https://api.diva-ai.cn/upload'
# 服务器根路径（用于音频文件下载）
SERVER_ROOT = 'http://localhost:3000' if IS_DEBUG else 'https://api.diva-ai.cn' if not IS_INTERNATIONAL else 'https://api.diva-ai.cn'
#密钥
API_KEY = 'diva-ai-model-scoring-2024'

# 更新文件下载接口
'''
@GET /client/update
@middleware api_key
@description 获取更新文件 结果以json返回
{
    "version": "1.0.0",
    "update_content": "更新内容",
    "update_file": "更新文件压缩包静态路径"
}
@return json
'''
UPDATE_FILE_URL = f'{API_URL}/client/update-file'

# 模型评分接口
MODEL_SCORING_URL = {
    # 模型待评分任务接口
    'pending_tasks': f'{API_URL}/model-scoring/pending-tasks',
    # 模型评分基础路径
    'base': f'{API_URL}/model-scoring',
    # 健康检查接口
    'health_check': f'{API_URL}/health'
}

# 模型评分配置
MODEL_SCORING_FETCH_INTERVAL = 3  # 获取任务间隔（秒）
MODEL_SCORING_MAX_CONCURRENT_TASKS = 5  # 最大并发任务数
MODEL_SCORING_TIMEOUT = 300  # 任务超时时间（秒）

# 模型评分脚本路径
#MODEL_SCORING_SCRIPT_DEBUG = '../model_scoring_client/inference_scores.py'
MODEL_SCORING_SCRIPT_DEBUG = "/home/zx/codexProject/vocal_analysis/Client/scripts/inference_scores.py"

# 非调试模式的评分脚本路径
#MODEL_SCORING_SCRIPT_PROD = 'D:/competition/inference_score_file.py'
MODEL_SCORING_SCRIPT_PROD = "/home/zx/codexProject/vocal_analysis/Client/scripts/inference_score_file.py"

# 根据调试模式选择脚本路径
MODEL_SCORING_SCRIPT = MODEL_SCORING_SCRIPT_DEBUG if IS_DEBUG else MODEL_SCORING_SCRIPT_PROD

# 在线评分任务拆分配置
SCORING_SPLIT_CONFIG = {
    # 最大并发拆分任务数
    'max_concurrent_splits': 3,
    
    # 输出基础目录
    'output_base_dir': 'outputs',
    
    # 临时下载目录
    'temp_download_dir': 'temp_downloads',
    
    # 音频拆分脚本路径（本地调试版本，10秒一段）
    #'splitting_script_path': 'D:\\competiton\\audio_process_pth.py',
    'splitting_script_path': '/home/zx/codexProject/vocal_analysis/Client/scripts/audio_process_pth.py',
    
    # 每次获取的待拆分任务数量
    'fetch_limit': 10,
    
    # 拆分完成后是否清理临时文件
    'cleanup_after_split': True,
    
    # 每段音频时长（秒）
    'segment_duration': 10
}

# 默认配置
DEFAULT_CONFIG = {
    'api_key': API_KEY,
    'base_url': API_URL,
    #'output_dir': 'C:\\Users\diva\\Desktop\\competition-diva-ai-main\\Client\outputs',
    'output_dir': '/home/zx/codexProject/vocal_analysis/Client/outputs',
    'max_concurrent_tasks': 3,
    'fetch_interval': 30,
    'batch_size': 10,
    'model_timeout': 300,
    'task_timeout': 300,
    'max_retries': 3,
    'retry_delay': 5
}







