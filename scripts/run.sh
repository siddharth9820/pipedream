#!/bin/bash

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            ARCH)               ARCH=${VALUE} ;;
            NUM_GPUS)           NUM_GPUS=${VALUE} ;;
            DATASET_NAME)       DATASET_NAME=${VALUE} ;;
            DATASET_DIR)        DATASET_DIR=${VALUE} ;;     
            *)   
    esac    
done

PIPEDREAM_HOME="/cfarhomes/ssingh37/pipedream"
BATCH_SIZE=64
RUN_PROFILE=1
RUN_OPTIM=1
if [ -z ${DATASET_DIR+x} ]; then SYNTHETIC_DATA=1; else SYNTHETIC_DATA=0; fi

MODE=1 #0 = Naive model parallelism, 1 = Hybrid parallelism, 2 = Data parallelism
MODEL_OUTPUT_DIR=$PIPEDREAM_HOME/runtime/image_classification/models/$DATASET_NAME/$ARCH/$NUM_GPUS
PROFILE_OUTPUT_DIR=$PIPEDREAM_HOME/profiler/image_classification/profiles/$DATASET_NAME

## STEP 1 - Run profiler
cd $PIPEDREAM_HOME/profiler/image_classification 
export LD_LIBRARY_PATH="/cfarhomes/ssingh37/miniconda3/lib/":$LD_LIBRARY_PATH

if [ $RUN_PROFILE -eq 1 ];then
    echo "Running Pipedream Profiler"
    if [ ! -d $PROFILE_OUPTPUT_DIR ]; then
        mkdir -p $PROFILE_OUTPUT_DIR
    fi 
    
    if [ $SYNTHETIC_DATA -eq 1 ];then
        echo "Using Synthetic Data.."
        python -u main_corrected.py -a $ARCH -b $BATCH_SIZE -s -v \
                                                        --profile_directory $PROFILE_OUTPUT_DIR 
                                                        
    else
        echo "Using $DATASET_NAME with $ARCH" 
        python -u main_corrected.py -a $ARCH -b $BATCH_SIZE --data_dir $DATASET_DIR -v \
                                                        --profile_directory $PROFILE_OUTPUT_DIR --dataset-name $DATASET_NAME
    fi
else 
    echo "Not Running Pipedream Profiler"
fi


cd $PIPEDREAM_HOME/optimizer

cat $PROFILE_OUTPUT_DIR/$ARCH/graph.txt
# STEP 2 - Run Optimizer
echo "Running PipeDream Optimizer"

if [ ! -d ./$DATASET_NAME/$ARCH/$NUM_GPUS ]; then
  mkdir -p ./$DATASET_NAME/$ARCH/$NUM_GPUS 
fi

if [ $RUN_OPTIM -eq 1 ];then
    python optimizer_graph_hierarchical.py -n $NUM_GPUS -f $PROFILE_OUTPUT_DIR/$ARCH/graph.txt -b 13000000000  -o ./$DATASET_NAME/$ARCH/$NUM_GPUS 

    repl_factors=$(<./$DATASET_NAME/$ARCH/$NUM_GPUS/stage_to_num_ranks_map.txt)


    if [ ! -d $MODEL_OUTPUT_DIR ]; then
        mkdir -p $MODEL_OUTPUT_DIR
    fi

    # ## STEP 3 - Convert output of optimizer into a pytorch model
    rm -rf $MODEL_OUTPUT_DIR/*
    python convert_graph_to_model.py -n $NUM_GPUS -f ./$DATASET_NAME/$ARCH/$NUM_GPUS/gpus=$NUM_GPUS.txt --stage_to_num_ranks_map $repl_factors -n $ARCH -a $ARCH -o $MODEL_OUTPUT_DIR
    cat $MODEL_OUTPUT_DIR/hybrid_conf.json

    echo $repl_factors
    
fi

repl_factors=$(<./$DATASET_NAME/$ARCH/$NUM_GPUS/stage_to_num_ranks_map.txt)
res="${repl_factors//[^:]}"
num_stages="${#res}"

# ## STEP 4 - Execute runtime
cd $PIPEDREAM_HOME/runtime/image_classification

# This is written strictly for all GPUs on a single node
if [ $MODE -eq 0 ]; then
    if [ $DATASET_NAME = "MNIST" ];then
        python -m torch.distributed.launch --nproc_per_node=$num_stages main_with_runtime.py --master_addr localhost --module models.$ARCH.$NUM_GPUS -b $BATCH_SIZE  --config_path $MODEL_OUTPUT_DIR/mp_conf.json --distributed_backend gloo --no_input_pipelining --dataset-name MNIST --data-dir $DATASET_DIR
    else 
        python -m torch.distributed.launch --nproc_per_node=$num_stages main_with_runtime.py --master_addr localhost --module models.$ARCH.$NUM_GPUS -b $BATCH_SIZE -s --config_path $MODEL_OUTPUT_DIR/mp_conf.json --distributed_backend gloo --no_input_pipelining 
    fi
fi 

if [ $MODE -eq 1 ]; then 
    if [ $SYNTHETIC_DATA -eq 0 ];then
        python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS main_with_runtime.py --master_addr localhost --module models.$DATASET_NAME.$ARCH.$NUM_GPUS -b $BATCH_SIZE --config_path $MODEL_OUTPUT_DIR/hybrid_conf.json --distributed_backend gloo --num_ranks_in_server $NUM_GPUS --dataset-name $DATASET_NAME  --data-dir $DATASET_DIR
    fi
fi

if [ $MODE -eq 2 ]; then 
    if [ $DATASET_NAME = "MNIST" ];then
        python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS main_with_runtime.py --master_addr localhost --module models.$ARCH.$NUM_GPUS -b $BATCH_SIZE -s --config_path $MODEL_OUTPUT_DIR/dp_conf.json --distributed_backend nccl --no_input_pipelining --dataset-name MNIST --data-dir $DATASET_DIR
    else
        python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS main_with_runtime.py --master_addr localhost --module models.$ARCH.$NUM_GPUS -b $BATCH_SIZE --config_path $MODEL_OUTPUT_DIR/dp_conf.json --distributed_backend nccl --no_input_pipelining
    fi
fi


cd $PIPEDREAM_HOME