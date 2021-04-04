ARGS="ARCH=vgg16 DATASET_NAME=ImageNet DATASET_DIR=/lus/theta-fs0/projects/CharmRTS/charmnn/imagenet/ RUN_PROFILE=0 CORRECT_OPTIM=0"
# 64  GPUs
qsub -t 500 -n 8 -A CharmRTS --mode script run.sh NUM_GPUS_PER_NODE=8 NUM_NODES=8 $ARCH
# 32  GPUs
qsub -t 500 -n 8 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=8 NUM_NODES=4 $ARCH
# 16  GPUs
qsub -t 500 -n 8 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=8 NUM_NODES=2 $ARCH
# 8  GPUs
qsub -t 500 -n 8 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=8 NUM_NODES=1 $ARCH
# 4  GPUs
qsub -t 500 -n 8 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=4 NUM_NODES=1 $ARCH
# 2  GPUs
qsub -t 500 -n 8 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=2 NUM_NODES=1 $ARCH
# 1  GPU
qsub -t 500 -n 8 -A CharmRTS --mode script  run.sh NUM_GPUS_PER_NODE=1 NUM_NODES=1 $ARCH
