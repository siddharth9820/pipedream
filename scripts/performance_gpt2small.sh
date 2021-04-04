ARGS="ARGS=gpt2_small DATASET_NAME=wikitext-103 LANG_MOD=1 DATASET_DIR=/lus/theta-fs0/projects/CharmRTS/charmnn/huggingface/ RUN_PROFILE=0 CORRECT_OPTIM=0"
# 64  GPUs
qsub -t 100 -n 8 -A CharmRTS --mode script run.sh NUM_GPUS_PER_NODE=8 NUM_NODES=8 $ARGS
# 32  GPUs
qsub -t 100 -n 4 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=8 NUM_NODES=4 $ARGS
# 16  GPUs
qsub -t 200 -n 2 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=8 NUM_NODES=2 $ARGS
# 8  GPUs
qsub -t 200 -n 1 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=8 NUM_NODES=1 $ARGS
# 4  GPUs
qsub -t 360 -n 1 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=4 NUM_NODES=1 $ARGS
# 2  GPUs
qsub -t 500 -n 1 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=2 NUM_NODES=1 $ARGS
# 1  GPU
qsub -t 500 -n 1 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=1 NUM_NODES=1 $ARGS
