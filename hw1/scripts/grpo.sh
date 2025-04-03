export HOST_IP=$MASTER_ADDR
export NCCL_CUMEM_ENABLE=0
export TOKENIZERS_PARALLELISM=False
export WANDB_KEY=$WANDB_KEY


# wandb setting
export WANDB_MODE=online
export WANDB_DIR=
ray stop


ROLLOUT_BS=64
N_SAMPLES_PER_PROMPT=8
TEMPERATURE=1.0
NUM_EPISODES=10
KL_COEF=0.01
BS=256
EP=1
LR=1e-6
EVAL_STEPS=10
MAX_GEN_LEN=1024

TRIAL_NAME=grpo_quality_round2

BASE_DIR=/mnt/data/zhuangyumin/cs2916-2025/hw1

DATA_PATH=${BASE_DIR}/data/train/math3k_rl_prompt2
# model path
POLICY_MODEL_PATH=${BASE_DIR}/../ckpts/grpo_quality_test1/global_step260_hf


SAVE_PATH=${BASE_DIR}/../ckpts/${TRIAL_NAME}
SAMPLES_SAVE_PATH=${BASE_DIR}/data/output/rl/${TRIAL_NAME}

# start rm
python -m src.cli.serve_rm \
    --mode rule \
    --tokenizer_path $POLICY_MODEL_PATH \
    --max_gen_len $MAX_GEN_LEN \
    --data_path $DATA_PATH \
    --port 5000 &

sleep 10s
RAY_MASTER_PORT=6379
RAY_DASHBOARD_PORT=8265
ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=127.0.0.1 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 4

sleep 10s
# replace working_dir with your own working dir
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit --address="http://127.0.0.1:12345" \
    --runtime-env-json="{\"working_dir\": \"${BASE_DIR}\", \"excludes\": [\"/data/\"]}" \
    -- python3 -m src.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --eval_steps $EVAL_STEPS \
    --save_steps 10 \
    --pretrain $POLICY_MODEL_PATH \
    --ref_pretrain $POLICY_MODEL_PATH \
    --remote_rm_url http://localhost:5000/get_reward \
    --save_path $SAVE_PATH \
    --ckpt_path $SAVE_PATH \
    --samples_save_path $SAMPLES_SAVE_PATH \
    --micro_train_batch_size 8 \
    --train_batch_size $BS \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size $ROLLOUT_BS \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --max_epochs $EP \
    --num_episodes $NUM_EPISODES \
    --prompt_max_len 350 \
    --generate_max_len $MAX_GEN_LEN \
    --advantage_estimator grpo \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate $LR \
    --lr_warmup_steps 10 \
    --init_kl_coef $KL_COEF \
    --prompt_data $DATA_PATH \
    --test_path ${BASE_DIR}/data/eval/RL.jsonl \
    --input_key context_messages \
    --apply_chat_template \
    --max_samples 100000 \
    --packing_samples \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --vllm_sync_backend nccl \
    --gradient_checkpointing \
    --temperature $TEMPERATURE \
    --use_wandb $WANDB_KEY \
    --wandb_org $WANDB_ORG \
    --wandb_project cs2916-2025 \
    --wandb_group rl.grpo \
    --wandb_run_name $TRIAL_NAME &
wait   
