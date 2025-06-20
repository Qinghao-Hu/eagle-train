
BASE_MODEL_PATH=${1}
MODEL_NAME=$(basename $BASE_MODEL_PATH)
DATA_PATH=/home/jerguo/dataset/datasets/tlt/eagle-mix

python freq_map/generate_freq.py \
    --model_name $MODEL_NAME \
    --model_path $BASE_MODEL_PATH \
    --dataset_path $DATA_PATH