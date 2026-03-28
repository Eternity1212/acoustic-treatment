# DPO 训练说明

该目录存放模块化拆分后的 DPO 声乐打分训练流程。推荐先修改
`config_dpo.json`，再通过 `train_dpo.py` 这个轻量入口启动训练。

## 文件说明

- `train_dpo.py`：命令行入口，负责读取 JSON 配置并启动训练
- `model_dpo.py`：`CAMPPlus` 模型及其依赖层定义
- `data_dpo.py`：MFCC、chosen、rejected 数据读取与数据集定义
- `trainer_dpo.py`：DPO loss、epoch 执行、checkpoint 与日志辅助逻辑
- `config_dpo.json`：默认路径和关键超参数配置文件

## 数据目录结构

数据集根目录下必须包含这些子目录：

```text
data_dir/
  MFCC_Output/
    sample_a_MFCC.xlsx
    sample_b_MFCC.xlsx
  Chosen/
    sample_a.xlsx
    sample_b.xlsx
  Rejected/
    sample_a.xlsx
    sample_b.xlsx
```

同一个 `sample_id` 必须同时出现在这三个位置。标签 Excel 需要能解析为
恰好 10 个标签，且原始标签值范围应为 `1..5`。

## 配置说明

训练前请先修改 `dpo/config_dpo.json`：

- `data_dir`：偏好数据集根目录
- `sft_checkpoint`：首次训练或 `policy_model_dir` 为空时，用于初始化 policy 的 SFT 模型路径
- `reference_checkpoint`：可选的冻结 reference 模型路径，留空时复用 `sft_checkpoint`
- `policy_model_dir`：固定 policy model 目录。若其中已有 `policy_model.pth`，后续 DPO 训练会优先从这里继续
- `output_dir`：DPO 训练输出目录根路径
- 其余训练与模型超参数

JSON 文件是主配置来源，少量命令行参数可以覆盖其中的字段。

## 启动方式

推荐在 `cbg` conda 环境中执行：

```bash
CONDA_NO_PLUGINS=true NUMBA_DISABLE_JIT=1 conda run -n cbg python3 dpo/train_dpo.py \
  --config dpo/config_dpo.json
```

常见覆盖参数示例：

```bash
CONDA_NO_PLUGINS=true NUMBA_DISABLE_JIT=1 conda run -n cbg python3 dpo/train_dpo.py \
  --config dpo/config_dpo.json \
  --beta 0.2 \
  --num-epochs 100 \
  --device cuda
```

## 输出内容

每次训练都会在 `output_dir/<timestamp>/` 下生成：

- `best_model.pth`
- `last_model.pth`
- `train.log`
- TensorBoard event files

此外，训练结束后当前的 policy model 还会同步保存到
`policy_model_dir/policy_model.pth`，供下一次 DPO 训练直接继续加载。
