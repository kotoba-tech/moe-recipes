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

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=61000  # 25,000 iteration = 100B Token

# optimizer config
LR=1e-4
MIN_LR=1e-6
LR_WARMUP_STEPS=2000
LR_DECAY_STEPS=61000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/model/MoE/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model
CHECKPOINT_DIR=/model/MoE/hf_checkpoints/Mixtral-8x1.3B-llm-jp
CHECKPOINT_SAVE_DIR="/model/MoE/checkpoint/Mixtral-8x1.3B/lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_seq_${SEQ_LENGTH}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} 16813821561 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_1.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16825021118 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_2.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16952900250 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_3.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16943850200 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_4.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16874407079 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_5.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16939269802 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_6.jsonl_text_document"
DATA_PATH="${DATA_PATH} 8483680205 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_7.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1444205466 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/ja_wiki_Llama2Tokenizer/ja_wiki_merge_1_text_document"
DATA_PATH="${DATA_PATH} 4704152329 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_wiki_Llama2Tokenizer/en_wiki_merge_1_text_document"
DATA_PATH="${DATA_PATH} 8673202783 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/code_stack_Llama2Tokenizer/code_stack_merge_1.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2015469612 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2013-2016.jsonl_text_document"
DATA_PATH="${DATA_PATH} 485206550 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-04.jsonl_text_document"
DATA_PATH="${DATA_PATH} 671309878.5 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-09.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1519299419 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-13.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2208918141 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-17.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1948758606 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-22.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2445789477 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-26.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1606556988 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-30.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2612392667 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-34.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1935587258 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-39.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2071044303 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-47.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1084445105 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-51.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1519525079 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-05.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2819466113 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-09.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1385647629 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-13.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1364184076 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-17.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1079338772 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-22.jsonl_text_document"
DATA_PATH="${DATA_PATH} 3152305557 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-26.jsonl_text_document"
DATA_PATH="${DATA_PATH} 3506323086 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-30.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2525217652 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-34.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2566432161 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-39.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2794133548 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-43.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2689865255 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-47.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2998236034 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-51.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2621878031 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-04.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2808203582 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-09.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2199366651 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-13.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2092968500 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-18.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2097900341 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-22.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1972868945 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-26.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1880666097 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-30.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2133914810 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-35.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1796919977 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-39.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1995263383 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-43.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1718482193 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-47.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1602519211 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-51.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2098557752 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-05.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2066051477 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-10.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1991951802 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-16.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2028506514 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-24.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2321431447 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-29.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1963744388 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-34.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2478089396 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-40.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1799373177 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-45.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1674274976 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-50.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2154557780 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-04.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1834506385 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-10.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2058310421 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-17.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1804132914 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-21.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1561087101 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-25.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1775614644 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-31.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1866858229 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-39.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2289643646 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-43.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2294948501 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-49.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2847207975 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-05.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2728433580 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-21.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2416260002 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-27.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2100352882 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-33.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1954450774 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-40.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2108624860 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-49.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2196011016 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-06.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1826237212 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-14.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1799114239 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-23.jsonl_text_document"

# job name
JOB_NAME="Mixtral-8x1.3B-llm-jp-cc-v2-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

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
  --hidden-size 2048 \
  --intermediate-size 7168 \
  --initializer-range 0.02 \
  --num-hidden-layers 24 \
  --num-attention-heads 16 \
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
