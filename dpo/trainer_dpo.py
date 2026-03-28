"""DPO 声乐打分训练的训练工具集合。

本模块负责 DPO loss、训练与验证 epoch、checkpoint 保存以及日志辅助逻辑，
让命令行入口保持精简并以配置驱动为主。
"""

import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SEED = 1314


def set_random_seed(seed):
    """为 python、numpy 和 torch 设置随机种子，便于复现实验。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reshape_logits(logits):
    """将扁平 logits 重排为 (batch, class, scoring_step) 结构。"""

    return logits.view(logits.shape[0], 5, 10)


def sequence_log_prob(logits, labels):
    """计算一条 10 步标签序列的联合对数概率。"""

    log_probs = F.log_softmax(logits.float(), dim=1)
    gathered = torch.gather(log_probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)
    return gathered.sum(dim=1)


def dpo_loss(policy_logits, reference_logits, chosen_labels, rejected_labels, beta):
    """使用整条标签序列对数概率计算标准 DPO 目标。"""

    policy_chosen = sequence_log_prob(policy_logits, chosen_labels)
    policy_rejected = sequence_log_prob(policy_logits, rejected_labels)
    reference_chosen = sequence_log_prob(reference_logits, chosen_labels)
    reference_rejected = sequence_log_prob(reference_logits, rejected_labels)

    pref_logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    loss = -F.logsigmoid(pref_logits).mean()

    metrics = {
        "loss": loss.item(),
        "preference_accuracy": (pref_logits > 0).float().mean().item(),
        "policy_margin": (policy_chosen - policy_rejected).mean().item(),
        "reference_margin": (reference_chosen - reference_rejected).mean().item(),
    }
    return loss, metrics


def move_batch_to_device(batch, device):
    """将一个 batch 中的张量字段移动到指定设备。"""

    mfcc, chosen, rejected, sample_ids = batch
    return mfcc.to(device), chosen.to(device), rejected.to(device), sample_ids


def run_epoch(policy_model, reference_model, data_loader, optimizer, device, beta, train_mode):
    """执行一轮训练或验证，并返回平均指标。"""

    policy_model.train(mode=train_mode)
    reference_model.eval()

    totals = {
        "loss": 0.0,
        "preference_accuracy": 0.0,
        "policy_margin": 0.0,
        "reference_margin": 0.0,
    }
    batch_count = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in data_loader:
            mfcc, chosen, rejected, _ = move_batch_to_device(batch, device)

            if train_mode:
                optimizer.zero_grad()

            policy_logits, _, _ = policy_model(mfcc)
            policy_logits = reshape_logits(policy_logits)

            reference_logits, _, _ = reference_model(mfcc)
            reference_logits = reshape_logits(reference_logits)

            loss, metrics = dpo_loss(policy_logits, reference_logits, chosen, rejected, beta)

            if train_mode:
                loss.backward()
                optimizer.step()

            for key in totals:
                totals[key] += metrics[key]
            batch_count += 1

    if batch_count == 0:
        raise ValueError("Data loader produced zero batches.")

    return {key: value / batch_count for key, value in totals.items()}


def load_model_weights(model, checkpoint_path, device):
    """将 checkpoint 加载到模型中，并兼容常见的包装键名。"""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(cleaned_state_dict)


def resolve_policy_checkpoint(policy_model_dir, fallback_checkpoint):
    """优先返回固定 policy 目录中的 checkpoint，不存在时回退到初始 checkpoint。"""

    policy_checkpoint = os.path.join(policy_model_dir, "policy_model.pth")
    if os.path.exists(policy_checkpoint):
        return policy_checkpoint
    return fallback_checkpoint


def save_state_dict(model, output_path):
    """保存当前 policy 模型的 state_dict。"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)


def create_run_dir(base_output_dir):
    """为一次 DPO 训练创建带时间戳的输出目录。"""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logging(run_dir):
    """为一次训练配置文件日志和控制台日志。"""

    log_file = os.path.join(run_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
