#!/bin/bash

set -e

start=15000
end=25000
increment=5000

upload_base_dir=/home/kazuki/converted_checkpoints/Mixtral-8x7b

# for ループで指定された範囲と増分を使用
for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/kotoba/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Mixtral-NVE-algebraic-stack-the-vault-lr_2e-5-min_lr_2e-6-iter$(printf "%07d" $i)
done
