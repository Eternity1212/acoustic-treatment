# 仓库智能体说明

本文件用于记录 `vocal_analysis` 仓库内与 DPO 开发相关的本地约束。

## DPO 约束

- DPO 代码必须放在 `dpo/` 目录中。为 DPO 流程开发时不要修改 `sft/`。
- 不要再把 DPO 训练回退成单个巨型脚本。入口文件保持精简，模型、数据和训练逻辑必须拆分到独立文件中。
- DPO 关键超参数统一维护在 `dpo/config_dpo.json` 中。命令行只作为少量覆盖层使用。
- 每个 DPO 源码文件开头都要写清晰的文件头注释或模块 docstring。
- 保持当前偏好数据格式约定：
  - `MFCC_Output/<sample_id>_MFCC.xlsx`
  - `Chosen/<sample_id>.xlsx`
  - `Rejected/<sample_id>.xlsx`

## 环境说明

- 优先使用 conda 的 `cbg` 环境运行。
- 在当前工作区中，执行 `conda` 命令时可能需要加上 `CONDA_NO_PLUGINS=true`。
- 当前环境下 `torchlibrosa/librosa` 在设置 `NUMBA_DISABLE_JIT=1` 时启动更稳定。

## 推荐启动命令

```bash
CONDA_NO_PLUGINS=true NUMBA_DISABLE_JIT=1 conda run -n cbg python3 dpo/train_dpo.py --config dpo/config_dpo.json
```
