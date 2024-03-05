#!/bin/bash
set -e

cd /home/taishi/moe-recipes
# swich virtual env
source .env/bin/activate

# ITERATION=500
# FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

# CHECK_POINT_DIR=/model/MoE/checkpoint/Mixtral-8x1.3B/lr_1e-4-minlr_1e-6_warmup_2000_seq_4096/${FORMATTED_ITERATION}

# python tools/checkpoint-convert/zero_to_fp32.py \
#   --checkpoint-dir $CHECK_POINT_DIR \
#   --output-file $CHECK_POINT_DIR/fp32_model.bin \
#   --debug

START=500
END=61000
STEP=500


for (( ITERATION=$START; ITERATION<=$END; ITERATION+=$STEP ))
do
    FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)
    CHECK_POINT_DIR=/model/MoE/checkpoint/Mixtral-8x1.3B/lr_1e-4-minlr_1e-6_warmup_2000_seq_4096/${FORMATTED_ITERATION}
    
    python tools/checkpoint-convert/zero_to_fp32.py \
      --checkpoint-dir $CHECK_POINT_DIR \
      --output-file $CHECK_POINT_DIR/fp32_model.bin \
      --debug

    echo "Conversion completed for $FORMATTED_ITERATION"
done