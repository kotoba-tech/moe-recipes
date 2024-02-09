#!/bin/bash

# swich virtual env
source .env/bin/activate

# distributed settings
JOB_ID=$(date +%s%N)
export MASTER_ADDR=10.130.184.10
export MASTER_PORT=12803

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="a100"

NHOSTS=16
NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

SGE_JOB_HOSTLIST=scripts/mdx/hostfile
export SGE_JOB_HOSTLIST

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# training config
# Mixtral-8x7B https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
SEQ_LENGTH=4096
SLIDING_WINDOW_SIZE=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000  # 25,000 iteration = 100B Token

# optimizer config
LR=5e-5
MIN_LR=1e-6
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/model/fujii/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model
CHECKPOINT_SAVE_DIR="/model/fujii/checkpoints/Mixtral-8x7B/lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_seq_${SEQ_LENGTH}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR="/model/fujii/datasets"

DATA_PATH=""

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} ${DATASET_DIR}/CC-MAIN-2013-2016.jsonl_text_document"


# job name
JOB_NAME="Mixtral-8x7B-llm-jp-cc-v2-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
  -bind-to none -map-by slot \
  -x PATH \
  python examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SLIDING_WINDOW_SIZE} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --lr-decay-iters ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-6 \
  --from-scratch \
  --hidden-size 4096 \
  --intermediate-size 14336 \
  --initializer-range 0.02 \
  --num-hidden-layers 32 \
  --num-attention-heads 32 \
  --num-key-value-heads 8 \
  --top-k 2 \
  --num-experts 8 \
  --router-aux-loss-coef 0.02 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --mixed-precision \
  --base-model "Mixtral" \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --use-zero \
  --zero-config "scripts/mdx/mixtral-config.json" \
  --zero-stage 3 \
  --no-meta-device \
  --output-router-logits \
  --use-mpi \
  --wandb-entity "llm-jp" \
  --wandb-project "Mixtral" \
  --wandb-name "${JOB_NAME}"
