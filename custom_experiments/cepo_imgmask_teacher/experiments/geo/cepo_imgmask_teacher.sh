#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
export CEPO_IS_PREFIX=0

MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_2b_geo_cepo_imgmask_teacher}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"
PROJECT_ROOT=/home/coder/lhc/CEPO
CUSTOM_ROOT=/home/coder/lhc/CEPO/custom_experiments/cepo_imgmask_teacher
export PYTHONPATH="${CUSTOM_ROOT}:${PYTHONPATH}"
MAX_STEPS=${MAX_STEPS:-50}
VAL_FREQ=${VAL_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

cd "${CUSTOM_ROOT}"

python3 -m verl.trainer.main \
    config=${PROJECT_ROOT}/examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.filter_overlong_prompts_workers=64 \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.max_prompt_length=768 \
    data.max_response_length=2048 \
    data.format_prompt=${PROJECT_ROOT}/examples/format_prompt/math_short.jinja \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=True \
    algorithm.use_kl_loss=False \
    algorithm.online_filtering=False \
    algorithm.cepo_use_cot_teacher=False \
    algorithm.cepo_cot_prefix_len=512 \
    algorithm.use_cepo=True \
    algorithm.cepo_lambda_init=0.5 \
    algorithm.cepo_warmup_steps=25 \
    algorithm.cepo_eps_w=0.5 \
    algorithm.cepo_lambda_schedule=linear \
    algorithm.cepo_teacher_mode=image_mask \
    algorithm.cepo_mask_mode=black \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=16 \
    worker.actor.optim.lr=5e-6 \
    worker.actor.global_batch_size=16 \
    worker.actor.use_torch_compile=False \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.fsdp.enable_full_shard=False \
    worker.actor.offload.offload_params=False \
    worker.actor.offload.offload_optimizer=False \
    worker.actor.model.enable_gradient_checkpointing=True \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.cepo_needs_ref=False \
    worker.rollout.n=8 \
    worker.ref.use_torch_compile=False \
    worker.ref.fsdp.enable_cpu_offload=False \
    worker.reward.reward_function=${PROJECT_ROOT}/examples/reward_function/math.py:compute_score \
    worker.rollout.enforce_eager=False \
    worker.rollout.max_model_len=2816 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.4 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.total_epochs=1 \
    trainer.max_steps=${MAX_STEPS} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="cepo" \
    trainer.val_freq=${VAL_FREQ} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.save_limit=2 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} 2>&1 | tee "${CUSTOM_ROOT}/logs/${LOGS_NAME}"