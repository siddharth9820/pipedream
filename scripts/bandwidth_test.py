import torch
from mpi4py import MPI
import argparse 
import time 
import torch.distributed as dist 
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument(
        '--gpus_per_node', help='number of GPUs per node', type=int, default=1)
parser.add_argument(
        '--backend', type=str, help='which backend', default='gloo'
)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--test_correctness', action='store_true', help='check for correctness')


def tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()

def to_MPI_buffer(tensor):
    return MPI.memory.fromaddress(tensor.data_ptr(), tensor_size(tensor))

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
local_rank = rank % args.gpus_per_node

#os.environ["CUDA_VISIBLE_DEVICES"]=f"{local_rank}"

print(f"Rank {rank}/{size} : Using GPU Number :- {local_rank}")

sz = 1*(2**30)
tensor = torch.empty([sz,])

print(tensor.shape, tensor.nelement(), tensor.element_size())

if args.backend == 'gloo':
    print("Testing gloo backend...")
    print("Testing P2P operations.. only supports synchronous communication between cpu tensors")
    tensor = tensor.cpu()   
    torch_comm_file = os.path.join(os.getcwd(), "temp.txt")
    if rank==0:
        if os.path.exists(torch_comm_file):
            os.remove(torch_comm_file)
    
    # os.environ["MASTER_ADDR"] = "thetagpu08"
    # os.environ["MASTER_PORT"] = "40000"
    
    dist.init_process_group('gloo', world_size=2, rank=rank, 
        init_method=f"file://{torch_comm_file}")

    print("Initialised process group")
    for at in range(20):
        if rank == 0:
            
            if args.test_correctness:
                torch.randn([sz,], out=tensor)
                rand = np.random.randint(5)
                ind = np.random.randint(100)
                tensor[ind] = rand

            st = time.time()
            
            dist.send(tensor=tensor, dst=1, tag=2*at)
            
            if args.test_correctness:
                tensor.zero_()
            
            wait_st = time.time()
            dist.recv(tensor=tensor, src=1, tag=2*at+1)
            wait_en = time.time()
            
            
            print("Time spent in receive call = ",wait_en-wait_st)
            
            if args.test_correctness:
                assert int(tensor[ind].item()) == rand
                print(f"Attempt {at}: Data was successfully received...")
            
            en = time.time()
            time_taken = en-st
            print(f"Bandwidth = {2*tensor_size(tensor)/time_taken/(1e9)} GBPS | Data Send {tensor_size(tensor)/1e9} GB")
            #time.sleep(2)
        else:
            dist.recv(tensor=tensor,src=0,tag=2*at)
            dist.send(tensor=tensor,dst=0,tag=2*at+1)
elif args.backend=='mpi':
    torch.cuda.set_device(local_rank)
    tensor = tensor.cuda()
    buffer = to_MPI_buffer(tensor)
    for at in range(10):
        if rank == 0:
            torch.randn([sz,], out=tensor)
            st = time.time()
            comm.Send([buffer, MPI.FLOAT], dest=1, tag=2*at)
            comm.Recv([buffer, MPI.FLOAT], source=1, tag=2*at+1)
            en = time.time()
            time_taken = en-st
            print(f"Bandwidth = {2*tensor_size(tensor)/time_taken/(2**30)} GBPS | Data Send {tensor_size(tensor)/2**30} GB")
            # print(copy[0])
            #time.sleep(2)
        else:
            comm.Recv([buffer, MPI.FLOAT], source=0, tag=2*at)
            comm.Send([buffer, MPI.FLOAT], dest=0, tag=2*at+1)
