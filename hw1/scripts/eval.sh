export HOST_IP=$MASTER_ADDR
export NCCL_CUMEM_ENABLE=0
export CUDA_VISIBLE_DEVICES=0

ckpt_path=../ckpts/sft_explore_512_2e-05/epoch_6
python src/evaluation/evaluation.py --model_path $ckpt_path --max_tokens 16384
