"""DPO 声乐打分训练的轻量级命令行入口。

本模块负责读取 JSON 配置、应用命令行覆盖参数，并组装模型、数据和训练器。
核心模型定义和训练逻辑不会继续堆在这个入口文件里。
"""

import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_dpo import PreferenceDataset
from model_dpo import build_model
from trainer_dpo import (
    create_run_dir,
    load_model_weights,
    run_epoch,
    save_state_dict,
    set_random_seed,
    setup_logging,
)


OVERRIDE_KEYS = [
    "data_dir",
    "sft_checkpoint",
    "reference_checkpoint",
    "output_dir",
    "device",
    "num_epochs",
    "learning_rate",
    "beta",
    "train_batch_size",
    "val_batch_size",
]


def parse_args():
    """解析配置文件路径以及可选的命令行覆盖参数。"""

    parser = argparse.ArgumentParser(description="训练用于声乐偏好优化的 DPO 模型。")
    parser.add_argument("--config", type=str, default="dpo/config_dpo.json", help="DPO JSON 配置文件路径。")
    parser.add_argument("--data-dir", type=str, default=None, help="覆盖数据集目录。")
    parser.add_argument("--sft-checkpoint", type=str, default=None, help="覆盖 SFT checkpoint 路径。")
    parser.add_argument(
        "--reference-checkpoint",
        type=str,
        default=None,
        help="覆盖 reference checkpoint 路径。留空时回退为 SFT checkpoint。",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="覆盖训练输出根目录。")
    parser.add_argument("--device", type=str, default=None, help="覆盖训练设备，例如 cpu 或 cuda。")
    parser.add_argument("--num-epochs", type=int, default=None, help="覆盖 DPO 训练轮数。")
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖优化器学习率。")
    parser.add_argument("--beta", type=float, default=None, help="覆盖 DPO 的 beta 系数。")
    parser.add_argument("--train-batch-size", type=int, default=None, help="覆盖训练 batch size。")
    parser.add_argument("--val-batch-size", type=int, default=None, help="覆盖验证 batch size。")
    return parser.parse_args()


def load_config(config_path):
    """读取 JSON 配置文件，并移除仅用于说明的注释字段。"""

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    return {key: value for key, value in config.items() if not key.startswith("_comment")}


def apply_overrides(config, args):
    """将非空命令行参数覆盖到 JSON 配置之上。"""

    merged = dict(config)
    for key in OVERRIDE_KEYS:
        arg_key = key.replace("-", "_")
        value = getattr(args, arg_key)
        if value is not None:
            merged[key] = value
    return merged


def validate_config(config):
    """在训练开始前检查必需配置项是否完整。"""

    required_keys = [
        "data_dir",
        "sft_checkpoint",
        "output_dir",
        "train_batch_size",
        "val_batch_size",
        "num_workers",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "beta",
        "seed",
        "device",
        "split_ratio",
        "num_classes",
        "input_size",
        "embd_dim",
        "growth_rate",
        "bn_size",
        "init_channels",
        "config_str",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    if not config.get("reference_checkpoint"):
        config["reference_checkpoint"] = config["sft_checkpoint"]
    return config


def main():
    """使用配置驱动方式启动一次 DPO 训练任务。"""

    args = parse_args()
    config = validate_config(apply_overrides(load_config(args.config), args))

    set_random_seed(config["seed"])
    run_dir = create_run_dir(config["output_dir"])
    setup_logging(run_dir)
    logging.info("Run directory: %s", run_dir)
    logging.info("Resolved config: %s", config)

    device = torch.device(config["device"])
    train_dataset = PreferenceDataset(
        data_dir=config["data_dir"],
        split="train",
        split_ratio=config["split_ratio"],
        seed=config["seed"],
    )
    val_dataset = PreferenceDataset(
        data_dir=config["data_dir"],
        split="val",
        split_ratio=config["split_ratio"],
        seed=config["seed"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    policy_model = build_model(config, device)
    reference_model = build_model(config, device)
    load_model_weights(policy_model, config["sft_checkpoint"], device)
    load_model_weights(reference_model, config["reference_checkpoint"], device)
    reference_model.requires_grad_(False)
    reference_model.eval()

    optimizer = torch.optim.Adam(
        policy_model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-7)
    writer = SummaryWriter(log_dir=run_dir)

    best_val_preference_accuracy = float("-inf")
    best_model_path = os.path.join(run_dir, "best_model.pth")
    last_model_path = os.path.join(run_dir, "last_model.pth")

    for epoch in range(config["num_epochs"]):
        train_metrics = run_epoch(
            policy_model=policy_model,
            reference_model=reference_model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=config["beta"],
            train_mode=True,
        )
        val_metrics = run_epoch(
            policy_model=policy_model,
            reference_model=reference_model,
            data_loader=val_loader,
            optimizer=None,
            device=device,
            beta=config["beta"],
            train_mode=False,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %d/%d | lr=%.8f | train_loss=%.4f | train_pref_acc=%.4f | val_loss=%.4f | val_pref_acc=%.4f",
            epoch + 1,
            config["num_epochs"],
            current_lr,
            train_metrics["loss"],
            train_metrics["preference_accuracy"],
            val_metrics["loss"],
            val_metrics["preference_accuracy"],
        )

        for split_name, metrics in [("Train", train_metrics), ("Val", val_metrics)]:
            writer.add_scalar(f"{split_name}/Loss", metrics["loss"], epoch)
            writer.add_scalar(f"{split_name}/PreferenceAccuracy", metrics["preference_accuracy"], epoch)
            writer.add_scalar(f"{split_name}/PolicyMargin", metrics["policy_margin"], epoch)
            writer.add_scalar(f"{split_name}/ReferenceMargin", metrics["reference_margin"], epoch)
        writer.add_scalar("Train/LearningRate", current_lr, epoch)

        save_state_dict(policy_model, last_model_path)
        if val_metrics["preference_accuracy"] > best_val_preference_accuracy:
            best_val_preference_accuracy = val_metrics["preference_accuracy"]
            save_state_dict(policy_model, best_model_path)
            logging.info(
                "Saved new best model with validation preference accuracy %.4f",
                best_val_preference_accuracy,
            )

    writer.close()
    logging.info("Training finished. Best validation preference accuracy: %.4f", best_val_preference_accuracy)


if __name__ == "__main__":
    main()
