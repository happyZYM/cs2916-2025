本次大作业基于`Qwen2.5-Math-1.5B`模型进行实验，自行完成的部分有：
- 给grpo部分补全reward计算和loss计算
- 修改sft和evalutaion的代码，使得可以高效地实验不同超参数对short cot sft的影响。

实验效果为：longcot sft、grpo均在4个指标上全序超过baseline；shortcot sft在2个指标上超过baseline，4个指标相对baseline比例的调和平均数为1.07，可以认为综合效果达到baseline。

# SFT
## short cot
实验效果：2个指标超过shortcot sft baseline，正确率相对baseline比值的调和平均数为1.07，可以认为综合效果达到baseline。  
最好的一次实验效果：
```
- AMC23 acc: 0.4
- GSM8k acc: 0.707
- MATH500 acc: 0.412
- OlympiadBench acc: 0.108
```
最好的一次实验对应的训练日志：<https://wandb.ai/zymx/cs2916-2025/runs/ju1v3as2>  
最好的一次实验训练参数：
```
BS=512
LR=2e-5
EP=6
```

## long cot
实验效果：所有指标全部超过longcot sft baseline
```
AMC23 acc: 0.325
GSM8k acc: 0.694
MATH500 acc: 0.408
OlympiadBench acc: 0.138
```
日志：<https://wandb.ai/zymx/cs2916-2025/runs/umndpqgo>  
训练参数：
```
BS=128
EP=6
LR=2e-5
```

# GRPO
实验效果：所有指标全部超过grpo baseline
```
AMC23 acc: 0.525
GSM8k acc: 0.775
MATH500 acc: 0.618
OlympiadBench acc: 0.246
```
## 训练记录
整个训练的流程是在GRPO的基础上进行人工划分阶段的curriculum learning。
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