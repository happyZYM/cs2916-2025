export HOST_IP=$MASTER_ADDR
export NCCL_CUMEM_ENABLE=0
export CUDA_VISIBLE_DEVICES=0

ckpt_path=../ckpts/sft_shortcot2
report_path=./data/output/report.jsonl
gpu_id=0

if [ $# -ge 1 ]; then
    ckpt_path=$1
fi

if [ $# -ge 2 ]; then
    report_path=$2
fi

if [ $# -ge 3 ]; then
    gpu_id=$3
    export CUDA_VISIBLE_DEVICES=$gpu_id
fi

echo "Evaluating model from $ckpt_path with report path $report_path on GPU $gpu_id"

python src/evaluation/evaluation_para.py --model_path $ckpt_path --max_tokens 16384 --report_path $report_path
