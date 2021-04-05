#!/bin/bash
source ~/.bashrc
RUN_PROFILE=1
LANG_MOD=0
CORRECT_OPTIM=0

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            ARCH)               ARCH=${VALUE} ;;
            NUM_GPUS_PER_NODE)  NUM_GPUS_PER_NODE=${VALUE} ;;
            NUM_NODES)          NUM_NODES=${VALUE} ;;
            DATASET_NAME)       DATASET_NAME=${VALUE} ;;
            DATASET_DIR)        DATASET_DIR=${VALUE} ;;
            RUN_PROFILE)        RUN_PROFILE=${VALUE} ;;
            LANG_MOD)           LANG_MOD=${VALUE} ;;
            CORRECT_OPTIM)      CORRECT_OPTIM=${VALUE} ;;
            *)   
    esac    
done

ARGS="--lr 0.1 -j 4"
USE_SSD=1
if [ $ARCH == "gpt2_small" ];then
    ARGS="--lr 0.001 --language_modelling -j 0 --bptt-len 256"
    BATCH_SIZE=32
    DATASET_DIR="/raid/scratch/huggingface"
elif [ $ARCH == "gpt2_medium" ];then 
    ARGS="--lr 0.001 --language_modelling -j 0 --bptt-len 128"
    BATCH_SIZE=32
    DATASET_DIR="/raid/scratch/huggingface"
elif [ $ARCH == "vgg16" ];then 
    ARGS="--lr 0.01  -j 4"
    BATCH_SIZE=64
    if [ $USE_SSD -eq 1 ];then
        export USE_SSD=1
        export TRAIN_TAR_FILE="$DATASET_DIR/ILSVRC2012_img_train.tar"
        export SSD_DIR="/raid/scratch"
        export DATASET_DIR=$DATASET_DIR
        mpirun -n $NUM_NODES -npernode 1 -x PATH -hostfile $COBALT_NODEFILE -bind-to none -map-by slot -x DATASET_DIR -x TRAIN_TAR_FILE -x SSD_DIR -x USE_SSD bash copy_to_ssd.sh
        DATASET_DIR=$SSD_DIR/imagenet
    fi
fi
    

PIPEDREAM_HOME="/home/ssingh37/pipedream"
RUN_OPTIM=1

## Change this part ##
INTRA_BW=200000000000
INTER_BW=25000000000
GLOO_SOCKET_IFNAME=infinibond0
IP=$(ip -4 addr show $GLOO_SOCKET_IFNAME | grep -oP "(?<=inet ).*(?=/)" 2>&1 | head -n 1)

intra=$(($INTRA_BW / 1000000000))
inter=$(($INTER_BW / 1000000000)) 

NUM_GPUS=$(($NUM_GPUS_PER_NODE*$NUM_NODES)) 

LOG_DIR="$PIPEDREAM_HOME/logs/$ARCH/gpus_$NUM_GPUS"


echo "Running on $NUM_NODES nodes, $NUM_GPUS_PER_NODE GPUs per node"
echo 
echo "============================================================"
echo "The following stuff is hardcoded (unfortunately!) in the bash script:"
echo "Bandwidths: (Intra Node) : $intra GBPS (Inter Node) : $inter GBPS"
echo "Using $GLOO_SOCKET_IFNAME IP - $IP socket for gloo P2P inter-node communication"
echo "If any of these values are wrong, please modify them in the bash script"
echo "============================================================"
echo



echo "Checking if dataset directory exists.."
if [ -z ${DATASET_DIR+x} ]; then SYNTHETIC_DATA=1; else SYNTHETIC_DATA=0; fi
if [ $SYNTHETIC_DATA -eq 1 ]; then
    echo "Dataset directory not found, switching to synthetic data.."
else 
    echo "Dataset directory found!"
fi


MODE=1 #0 = Naive model parallelism, 1 = Hybrid parallelism, 2 = Data parallelism

MODEL_OUTPUT_DIR=$PIPEDREAM_HOME/runtime/image_classification/models/$DATASET_NAME/$ARCH/$NUM_GPUS
PROFILE_OUTPUT_DIR=$PIPEDREAM_HOME/profiler/image_classification/profiles/$DATASET_NAME

echo

## STEP 1 - Run profiler
cd $PIPEDREAM_HOME/profiler/image_classification 
export LD_LIBRARY_PATH=$CONDA_DIR:$LD_LIBRARY_PATH

if [ $RUN_PROFILE -eq 1 ];then
    echo "Running Pipedream Profiler"
    if [ ! -d $PROFILE_OUPTPUT_DIR/$ARCH ]; then
        mkdir -p $PROFILE_OUTPUT_DIR/$ARCH
    fi 
 
    echo "Using $DATASET_NAME with $ARCH"
    python -u main_corrected.py -a $ARCH -b $BATCH_SIZE -s -v --profile_directory $PROFILE_OUTPUT_DIR --dataset-name $DATASET_NAME 
  
else 
    echo "Not Running Pipedream Profiler"
fi


cd $PIPEDREAM_HOME/optimizer

cat $PROFILE_OUTPUT_DIR/$ARCH/graph.txt
# STEP 2 - Run Optimizer

if [ ! -d ./$DATASET_NAME/$ARCH/$NUM_GPUS ]; then
  mkdir -p ./$DATASET_NAME/$ARCH/$NUM_GPUS 
fi

if [ $RUN_OPTIM -eq 1 ];then
    echo "Running PipeDream Optimizer and generating pipedream model"

    if [ $CORRECT_OPTIM -eq 0 ];then
        echo "Using original optimizer"
        python optimizer_graph_hierarchical.py -n $NUM_GPUS_PER_NODE $NUM_NODES -f $PROFILE_OUTPUT_DIR/$ARCH/graph.txt -b $INTRA_BW $INTER_BW  -o ./$DATASET_NAME/$ARCH/$NUM_GPUS
    else 
        echo "Using corrected optimizer"
        python optimizer_graph_hierarchical_modified.py -n $NUM_GPUS_PER_NODE $NUM_NODES -f $PROFILE_OUTPUT_DIR/$ARCH/graph.txt -b $INTRA_BW $INTER_BW  -o ./$DATASET_NAME/$ARCH/$NUM_GPUS
    fi 

    repl_factors=$(<./$DATASET_NAME/$ARCH/$NUM_GPUS/stage_to_num_ranks_map.txt)

    if [ ! -d $MODEL_OUTPUT_DIR ]; then
        mkdir -p $MODEL_OUTPUT_DIR
    fi

    # ## STEP 3 - Convert output of optimizer into a pytorch model
    rm -rf $MODEL_OUTPUT_DIR/*

    echo "generating model"
    python convert_graph_to_model.py -n $NUM_GPUS -f ./$DATASET_NAME/$ARCH/$NUM_GPUS/gpus=$NUM_GPUS.txt --stage_to_num_ranks_map $repl_factors -n $ARCH -a $ARCH -o $MODEL_OUTPUT_DIR
    
    
fi

if [ ! -f $MODEL_OUTPUT_DIR/hybrid_conf.json ]; then
    echo "Detected pure data parallelism configuration"
    MODE=2
fi

## STEP 4 - Execute runtime
cd $PIPEDREAM_HOME/runtime/image_classification


if [ -d $LOG_DIR ]; then
  echo "Deleting $LOG_DIR"
  rm -rf $LOG_DIR 
fi

echo "Creating $LOG_DIR"
mkdir -p $LOG_DIR



if [ $MODE -eq 1 ]; then     
    echo "launching hybrid parallelism job"
    
    if [ -f ./temp.txt ]; then
        echo "deleting torch comm file.."
        rm ./temp.txt
    fi 

    mpirun -npernode $NUM_GPUS_PER_NODE --cpus-per-proc 16 -n $NUM_GPUS -x PATH -x GLOO_SOCKET_IFNAME=infinibond0 -hostfile $COBALT_NODEFILE python main_with_runtime.py --master_addr $IP --module models.$DATASET_NAME.$ARCH.$NUM_GPUS -b $BATCH_SIZE --config_path $MODEL_OUTPUT_DIR/hybrid_conf.json --distributed_backend gloo --dataset-name $DATASET_NAME  --data-dir $DATASET_DIR  --num_ranks_in_server $NUM_GPUS_PER_NODE --world_size $NUM_GPUS --log_dir $LOG_DIR --print-freq 10 $ARGS 
fi

if [ $MODE -eq 2 ]; then 
    echo "launching data parallel job"

    if [ -f ./temp.txt ]; then
        echo "deleting torch comm file.."
        rm ./temp.txt
    fi 

    mpirun -npernode $NUM_GPUS_PER_NODE --cpus-per-proc 16 -n $NUM_GPUS -x PATH -x NCCL_SOCKET_IFNAME=infinibond0 -x GLOO_SOCKET_IFNAME=infinibond0 -hostfile $COBALT_NODEFILE python main_with_runtime.py --master_addr $IP --module models.$DATASET_NAME.$ARCH.$NUM_GPUS -b $BATCH_SIZE --config_path $MODEL_OUTPUT_DIR/dp_conf.json --distributed_backend nccl --no_input_pipelining --dataset-name $DATASET_NAME  --data-dir $DATASET_DIR --data_prl --world_size $NUM_GPUS --num_ranks_in_server $NUM_GPUS_PER_NODE --log_dir $LOG_DIR --print-freq 10 --world_size $NUM_GPUS $ARGS 
fi


cd $PIPEDREAM_HOME
