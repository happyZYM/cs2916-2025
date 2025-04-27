# SFT
## short cot

## long cot


# RL

## 训练记录
### 基本训练
本阶段使用默认数据集、默认LOSS函数，reward函数如下：
- 回答无法解析出答案：`-1`
- 答案错误：`0`
- 答案正确：`100`

训练日志
- round1: <https://wandb.ai/zymx/cs2916-2025/runs/goiyu708>，从`Qwen2.5-Math-1.5B`得到模型`grpo_test1/global_step120_hf`
- round2: <https://wandb.ai/zymx/cs2916-2025/runs/etkl63s3>，从上一模型得到模型`grpo_test2/global_step30_hf`
- round3: <https://wandb.ai/zymx/cs2916-2025/runs/ofi0fjo8>，从上一模型得到模型`grpo_test3/global_step40_hf`

### 质量提升训练
本阶段使用默认数据集、默认LOSS函数，reward函数如下：
- 回答无法解析出答案：`-10`
- 答案错误：`0`
- 答案正确：`100`
- 思维链长度bonus：在回答中没有重复的情况下，提供一个不超过10的长度bonus
- 语言bonus：不出现中英混杂的情况时，可获得`5`分的bonus

训练日志：
- round1：<https://wandb.ai/zymx/cs2916-2025/runs/4zu1r8tm>，从上一模型得到模型`grpo_quality_test1/global_step260_hf`
- round2: <https://wandb.ai/zymx/cs2916-2025/runs/56h3izw8>，从上一模型使用Clip-Higher得到`grpo_quality_round2/global_step110_hf`

# RL2
longcot5 -> grpo_quality_l6/global_step70_hf -> grpo_quality_l7/global_step450_hf