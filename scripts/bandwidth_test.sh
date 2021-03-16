#!/bin/bash
source ~/.bashrc 
source $SYS_CONDA


cat $COBALT_NODEFILE | grep -v $(/bin/hostname -s)

PORT=20000
RANK=1
for node in $WORKERS; do
    ssh -q $node "source ~/.bashrc; source $SYS_CONDA; python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=$RANK --master_addr=thetagpu12 --master_port=$PORT bandwidth_test.py" &
    RANK=$((RANK+1))
done

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=thetagpu12 --master_port=$PORT bandwidth_test.py 

# ssh -q thetagpu12 "source ~/.bashrc; source $SYS_CONDA; cd ~/pipedream/scripts; python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=thetagpu12 --master_port=10000 bandwidth_test.py" &