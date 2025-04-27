import os
import json
import jsonlines
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import time

max_explore_epoch = 6
evaluation_epochs = [3, 6]
explore_batch_size = [128, 256, 512]
explore_lr = [1e-5, 2e-5, 4e-5]
report_path = './data/output/explore_report.jsonl'
report_data = {}

max_evaluation_concurrency = 4
available_gpus = [0, 1, 2, 3]  # 可用的GPU ID列表

# 使用Manager创建一个在进程间共享的字典来跟踪GPU状态
manager = multiprocessing.Manager()
gpu_status = manager.dict({gpu_id: False for gpu_id in available_gpus})  # False表示GPU空闲
gpu_lock = manager.Lock()  # 添加一个锁来保护GPU状态访问

def get_gpu(gpu_status_dict, lock):
    """获取一个空闲的GPU"""
    with lock:  # 使用锁保护临界区
        for gpu_id in available_gpus:
            if not gpu_status_dict[gpu_id]:  # 如果GPU空闲
                gpu_status_dict[gpu_id] = True  # 标记为已占用
                return gpu_id
    return None  # 没有空闲GPU

def release_gpu(gpu_id, gpu_status_dict, lock):
    """释放一个GPU"""
    with lock:  # 使用锁保护临界区
        gpu_status_dict[gpu_id] = False  # 标记为空闲

def evaluate_model(bs, lr, epoch, gpu_status_shared, lock):
    try:
        # 等待获取可用GPU
        gpu_id = None
        while gpu_id is None:
            gpu_id = get_gpu(gpu_status_shared, lock)
            if gpu_id is None:
                print(f"任务 (bs={bs}, lr={lr}, epoch={epoch}) 等待GPU...")
                time.sleep(5)  # 等待5秒后重试
        
        print(f"任务 (bs={bs}, lr={lr}, epoch={epoch}) 分配到GPU: {gpu_id}")
        
        eval_report_path = f"./data/output/eval_report_{bs}_{lr}_{epoch}.jsonl"
        eval_command = f"scripts/eval_para.sh ../ckpts/sft_explore_{bs}_{lr}/epoch_{epoch} {eval_report_path} {gpu_id}"
        print(f"Running on GPU {gpu_id}: {eval_command}")
        os.system(eval_command)
        
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            eval_report = json.load(f)
        
        # 释放GPU
        release_gpu(gpu_id, gpu_status_shared, lock)
        print(f"任务 (bs={bs}, lr={lr}, epoch={epoch}) 释放GPU: {gpu_id}")
        
        return (bs, lr, epoch), eval_report
    except Exception as e:
        print(f"评估模型出错: {e}")
        # 确保释放GPU
        if 'gpu_id' in locals() and gpu_id is not None:
            release_gpu(gpu_id, gpu_status_shared, lock)
            print(f"错误处理: 任务 (bs={bs}, lr={lr}, epoch={epoch}) 释放GPU: {gpu_id}")
        return (bs, lr, epoch), None

# 收集需要评估的任务
evaluation_tasks = []

for bs in explore_batch_size:
    for lr in explore_lr:
        training_command = f"scripts/sft_cot_para.sh {bs} {max_explore_epoch} {lr} sft_explore_{bs}_{lr}"
        print(training_command)
        os.system(training_command)
        for eval_epoch in evaluation_epochs:
            evaluation_tasks.append((bs, lr, eval_epoch))

# 并行执行评估任务
with ProcessPoolExecutor(max_workers=max_evaluation_concurrency) as executor:
    futures = []
    for bs, lr, epoch in evaluation_tasks:
        futures.append(executor.submit(evaluate_model, bs, lr, epoch, gpu_status, gpu_lock))
    
    for future in futures:
        key, eval_report = future.result()
        if eval_report is not None:
            report_data[key] = eval_report
            print(f"Report data: {report_data}")
        else:
            print(f"任务 {key} 评估失败")

with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report_data, f, ensure_ascii=False)
