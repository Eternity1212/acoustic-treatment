import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import API_URL, API_KEY

# 模型评分任务获取速率 单位 秒
MODEL_SCORING_FETCH_INTERVAL = 3

# 模型评分接口
MODEL_SCORING_URL = {
    '''
    获取模型待评分任务
    @GET /model-scoring/pending-tasks
    @return 
    '''
    # 模型待评分任务接口
    'pending_tasks': f'{API_URL}/model-scoring/pending-tasks',

    '''
    提交模型评分
    @POST /model-scoring
    @param {string} taskId - 任务ID
    @param {number} score - 模型评分
    @return 
    '''
    # 模型评分提交接口
    'submit_score': f'{API_URL}/model-scoring',
    
    '''
    更新任务状态
    @PUT /model-scoring/task/{task_id}/status
    @param {string} task_id - 任务ID
    @param {string} status - 状态
    @return 
    '''
    # 更新任务状态接口
    'update_status': f'{API_URL}/model-scoring/task',
    
    '''
    健康检查
    @GET /health
    @return 
    '''
    # 健康检查接口
    'health_check': f'{API_URL}/health'
}

# 模型评分脚本路径
MODEL_SCORING_SCRIPT = '../model_scoring_client/inference_scores.py'

# 默认配置
DEFAULT_CONFIG = {
    'api_key': API_KEY,
    'base_url': API_URL,
    'output_dir': 'C:/Users/diva/Desktop/competition-diva-ai-main/Client/outputs',
    'max_concurrent_tasks': 3,
    'fetch_interval': 30,
    'batch_size': 10,
    'model_timeout': 300,
    'task_timeout': 300,
    'max_retries': 3,
    'retry_delay': 5
}
