import os
import json
import numpy as np
import matplotlib.pyplot as plt

report_path = './data/output'

evaluation_epochs = [3, 6]
explore_batch_size = [128, 256, 512]
explore_lr = [1e-5, 2e-5, 4e-5]

baseline = {
    "AMC23": 0.3,
    "GSM8k": 0.716,
    "MATH500": 0.44,
    "OlympiadBench": 0.096
}

def get_overall_score(data):
    score_list = []
    for key, value in data.items():
        if key in baseline:
            score_list.append(value/baseline[key])
    # 使用调和平均数增大最小值的影响
    score_array = np.array(score_list)
    return len(score_array) / np.sum(1.0 / score_array)

benchmark_data = {}

for bs in explore_batch_size:
    for lr in explore_lr:
        for epoch in evaluation_epochs:
            file_path = f'{report_path}/eval_report_{bs}_{lr}_{epoch}.jsonl'
            print(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(data)
            key = (bs, lr, epoch)
            benchmark_data[key] = get_overall_score(data['acc'])

print(benchmark_data)
# 打印最大值
max_score = max(benchmark_data.values())
max_key = max(benchmark_data, key=benchmark_data.get)
print(f"最大值: {max_score}, 对应的参数: {max_key}")

# 可视化结果
plt.figure(figsize=(15, 10))

# 1. 热力图展示不同batch_size和learning_rate在不同epoch下的得分
for i, epoch in enumerate(evaluation_epochs):
    data_matrix = np.zeros((len(explore_batch_size), len(explore_lr)))
    for i_bs, bs in enumerate(explore_batch_size):
        for j_lr, lr in enumerate(explore_lr):
            data_matrix[i_bs, j_lr] = benchmark_data.get((bs, lr, epoch), 0)
    
    plt.subplot(2, 2, i+1)  # Fixed: using index+1 instead of epoch value
    im = plt.imshow(data_matrix, cmap='viridis')
    plt.colorbar(im, label='Score')
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    plt.title(f'Model Score at Epoch {epoch}')
    plt.xticks(np.arange(len(explore_lr)), [f"{lr:.0e}" for lr in explore_lr])
    plt.yticks(np.arange(len(explore_batch_size)), explore_batch_size)
    
    # Add value annotations on heatmap
    for i_bs in range(len(explore_batch_size)):
        for j_lr in range(len(explore_lr)):
            value = data_matrix[i_bs, j_lr]
            plt.text(j_lr, i_bs, f"{value:.3f}", ha="center", va="center", 
                     color="white" if value > np.mean(data_matrix) else "black")

# 2. 条形图比较所有组合的得分
plt.subplot(2, 2, 3)
labels = [f"bs={k[0]},lr={k[1]:.0e},e={k[2]}" for k in benchmark_data.keys()]
values = list(benchmark_data.values())
plt.bar(range(len(labels)), values, color='skyblue')
plt.xticks(range(len(labels)), labels, rotation=90)
plt.ylabel('Score')
plt.title('Model Score for All Parameter Combinations')

# 标记最佳组合
best_idx = list(benchmark_data.values()).index(max_score)
plt.bar(best_idx, max_score, color='red')

# 3. 不同参数对得分的影响折线图
plt.subplot(2, 2, 4)

# 按batch_size分组
bs_scores = {}
for bs in explore_batch_size:
    bs_scores[bs] = [benchmark_data.get((bs, lr, epoch), 0) 
                    for lr in explore_lr for epoch in evaluation_epochs]

for bs, scores in bs_scores.items():
    plt.plot(range(len(scores)), scores, label=f'bs={bs}', marker='o')

plt.xlabel('Parameter Combination Index')
plt.ylabel('Score')
plt.title('Score Comparison Across Different Batch Sizes')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=300)
plt.show()

# 额外生成一个图表展示每个数据集的准确率与基准线的对比
# 获取最佳参数组合下的数据
best_bs, best_lr, best_epoch = max_key
best_file_path = f'{report_path}/eval_report_{best_bs}_{best_lr}_{best_epoch}.jsonl'
with open(best_file_path, 'r', encoding='utf-8') as f:
    best_data = json.load(f)

plt.figure(figsize=(10, 6))
benchmarks = list(baseline.keys())
best_accs = [best_data['acc'].get(b, 0) for b in benchmarks]
baseline_accs = [baseline[b] for b in benchmarks]

x = np.arange(len(benchmarks))
width = 0.35

plt.bar(x - width/2, baseline_accs, width, label='Baseline', color='lightgray')
plt.bar(x + width/2, best_accs, width, label=f'Best Model (bs={best_bs},lr={best_lr:.0e},e={best_epoch})', color='lightgreen')

# 标记每个柱状图的具体数值
for i, v in enumerate(baseline_accs):
    plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
for i, v in enumerate(best_accs):
    plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
    
plt.xlabel('Benchmark Dataset')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Best Model vs Baseline')
plt.xticks(x, benchmarks)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('best_model_comparison.png', dpi=300)
plt.show()
