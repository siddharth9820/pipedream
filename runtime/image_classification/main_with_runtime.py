# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time

import torch
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import socket
import fcntl
import struct

def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])



sys.path.append("..")
sys.path.append("../../dataloaders/language_modelling")


import runtime
import sgd
import adam 
import transformers

import huggingface

from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-dir', type=str,
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--lr_warmup', action='store_true',
                    help='Warmup learning rate first 5 epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs",default=False)
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--world_size', default=0, type=int,
                    help="Total number of gpus")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")
# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')
parser.add_argument('--dataset-name', default='ImageNet', type=str,
                    help='distributed backend')
parser.add_argument('--bptt-len', default=256, type=int,
                    help='Backpropagation through time length for Language Modelling tasks (ignored for Image Classification)')

parser.add_argument('--language_modelling', action='store_true',
                    help="Run a Language Modelling Task", default=False)

parser.add_argument('--data_prl', action='store_true',
                    help='data parallelism')

parser.add_argument('--log_dir', help='directory to save logs into', type=str)
best_prec1 = 0



# Helper methods.
def is_first_stage():
    return args.stage is None or (args.stage == 0)

def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))


def print_msg(rank, msg):
    print("Rank {} - {}".format(rank, msg))

# Synthetic Dataset class.
class SyntheticDatasetLanguageModelling(torch.utils.data.dataset.Dataset):
    def __init__(self, vocab_size, bptt_len, length):
        self.length = length 
        sentence = torch.randint(low=0, high=vocab_size, size=(bptt_len,)).long()
        self.src = sentence[:-1]
        self.trg = sentence [1:]

    def __getitem__(self, index):
        return self.src, self.trg

    def __len__(self):
        return self.length

class SyntheticDatasetImageClassification(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length


def main():
    global args, best_prec1
    args = parser.parse_args()
    if rank >= args.world_size:
        return
    print("initialising device...")
    local_rank = rank % args.num_ranks_in_server
    print("workers = ",args.workers)

    writer=None
    if args.log_dir:
        writer=SummaryWriter(log_dir=args.log_dir)
    ##### ENABLING GPU DIRECT HERE THROUGH A HACK ###
    args.num_ranks_in_server = args.world_size

    torch.cuda.set_device(local_rank)
    print("local rank {} device {}".format(local_rank, torch.cuda.current_device()))
    args.rank = rank # my change
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    model = module.model(criterion)
    print("Local rank {} imported module".format(local_rank))

    # determine shapes of all tensors in passed-in model
    
    target_size = [args.batch_size]
    if args.dataset_name == "ImageNet":
        if args.arch == 'inception_v3':
            input_size = [args.batch_size, 3, 299, 299]
        else:
            input_size = [args.batch_size, 3, 224, 224]
        first_stage_input_dtype = torch.float32
    elif args.dataset_name == "MNIST":
        input_size = [args.batch_size, 1, 28, 28]
        first_stage_input_dtype = torch.float32
    elif args.dataset_name == "CIFAR10":
        input_size = [args.batch_size, 3, 32, 32]
        first_stage_input_dtype = torch.float32
    elif args.dataset_name in ["wikitext-2", "wikitext-103"]:
        input_size = [args.batch_size, args.bptt_len]
        first_stage_input_dtype = torch.int64
        target_size = [args.batch_size*args.bptt_len]
    else:
        print("Dataset {} not supported".format(args.dataset_name))

    training_tensor_shapes = {"input0": input_size, "target": target_size}
    dtypes = {"input0": torch.int64, "target": torch.int64}
    inputs_module_destinations = {"input": 0}
    target_tensor_names = {"target"}
    stage_number = 0
    for (stage, inputs, outputs) in model[:-1]:  # Skip last layer (loss).
        input_tensors = []
        for input in inputs:
            if stage_number == 0:
                input_dtype = first_stage_input_dtype
            else:
                input_dtype = torch.float32
            
            input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
                                       dtype=input_dtype).cuda()
                                    
            input_tensors.append(input_tensor)
        stage_number += 1
        stage.cuda()
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype
        del output_tensors
        del input_tensors
        stage.cpu()

    #print("local rank {} finished 1 forward pass...".format(local_rank))
    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            [args.eval_batch_size] + training_tensor_shapes[key][1:])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        # print("========================")
        # print(configuration_maps['stage_to_rank_map'])
        
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)
    
    if args.data_prl:
        print("Modifying stage to rank map to be data parallel")
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        for k in stage_to_rank_map:
            stage_to_rank_map[k] = list(range(args.world_size))

    print("Local rank {} Staging runtime....".format(local_rank))

    if args.language_modelling:
        model_type = runtime.LANGUAGE_MODELLING
    else:
        model_type = runtime.IMAGE_CLASSIFICATION


    r = runtime.StageRuntime(
        model=model, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        local_rank=local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=model_type,
        enable_recompute=args.recompute)

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not is_first_stage():
        args.synthetic_data = True

    # define optimizer
    if args.no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        num_versions = r.num_warmup_minibatches + 1

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_file_path, checkpoint['epoch']))

    #print_msg(args.rank, "number of versions" + str(num_versions) )

    if args.language_modelling:
        optimizer = adam.AdamWithWeightStashing(r.modules(),r.master_parameters, r.model_parameters, args.loss_scale, num_versions=num_versions, lr=args.lr, weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency, macrobatch=args.macrobatch)
    else:
        # optimizer = sgd.SGDWithWeightStashing(r.modules(), r.master_parameters,
        #                                   r.model_parameters, args.loss_scale,
        #                                   num_versions=num_versions,
        #                                   lr=args.lr,
        #                                   momentum=args.momentum,
        #                                   weight_decay=args.weight_decay,
        #                                   verbose_freq=args.verbose_frequency,
        #                                   macrobatch=args.macrobatch)
        optimizer = adam.AdamWithWeightStashing(r.modules(),r.master_parameters, r.model_parameters, args.loss_scale, num_versions=num_versions, lr=args.lr, weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency, macrobatch=args.macrobatch)
    

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    print(args.dataset_name)
    

    if args.dataset_name == "ImageNet":
        if args.arch == 'inception_v3':
            if args.synthetic_data:
                train_dataset = SyntheticDatasetImageClassification((3, 299, 299), 10000)
            else:
                traindir = os.path.join(args.data_dir, 'train')
                train_dataset = datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.RandomResizedCrop(299),
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
        else:
            print("Initialising dataset..")
            if args.synthetic_data:
                train_dataset = SyntheticDatasetImageClassification((3, 224, 224),  1281168 ) #modified
            else:
                traindir = os.path.join(args.data_dir, 'train')
                train_dataset = datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]))

        if args.synthetic_data:
            val_dataset = SyntheticDatasetImageClassification((3, 224, 224), 10000)
        else:
            # valdir = os.path.join(args.data_dir, 'val')
            # val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.CenterCrop(224),
            #     transforms.ToTensor(),
            #     normalize,
            # ]))
            val_dataset = SyntheticDatasetImageClassification((3, 224, 224), 10000)
    
    elif args.dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            args.data_dir,
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)),
            ]))

        val_dataset = datasets.MNIST(args.data_dir, download=True,
                                                    train=False,
                                                    transform=transforms.Compose([
                                                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                                    transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))
    elif args.dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform =transform)
        val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform =transform)
    elif args.dataset_name in args.dataset_name in ["wikitext-2", "wikitext-103"]:
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        if not args.synthetic_data:
            train_dataset = huggingface.get_dataset(args.dataset_name, tokenizer, 'train', num_workers=1, bptt_len=args.bptt_len, cache_dir=args.data_dir)
            val_dataset = huggingface.get_dataset(args.dataset_name, tokenizer, 'validation', num_workers=1, bptt_len=args.bptt_len, cache_dir=args.data_dir)
        else:
            if args.dataset_name == "wikitext-2":
                train_length = 36718
            else:
                train_length = 1801350
            train_dataset = SyntheticDatasetLanguageModelling(tokenizer.vocab_size, args.bptt_len, train_length)
            val_dataset = SyntheticDatasetLanguageModelling(tokenizer.vocab_size, args.bptt_len, 3760)

    distributed_sampler = False
    train_sampler = None
    val_sampler = None
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank%num_ranks_in_first_stage)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank%num_ranks_in_first_stage)
            distributed_sampler = True

    print("initialising data loaders")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    print(f"Rank {args.rank}: Length of train loader: {len(train_loader)} Length of dataset: {len(train_dataset)} BPTT_LEN {args.bptt_len} BATCH SIZE {args.batch_size}")
    # else:
    #     train_loader = None
    #     val_loader = None 
    
    # if args.rank==0:
    #     lengths = torch.LongTensor([len(train_loader), len(val_loader)]).cuda()
    # else:
    #     lengths = torch.zeros((2)).long().cuda()
    # dist.broadcast(lengths, src=0)

    lengths = torch.LongTensor([len(train_loader), len(val_loader)])

    quantities = [len(configuration_maps['stage_to_rank_map'][0])]
    for i in range(len(configuration_maps['stage_to_rank_map']) - 1):
        curr = len(configuration_maps['stage_to_rank_map'][i])
        curr *= len(configuration_maps['stage_to_rank_map'][i+1])
        quantities.append(curr)
    print(quantities)
    lcm = np.lcm.reduce(quantities)
    print(f"new length should be a multiple of {lcm}")
    old_length = lengths[0].item()
    lengths[0] = (lengths[0] // lcm) * lcm
    print(f"Old length {old_length} Adjusted Length {lengths[0]}")

    #exit()
    # if checkpoint is loaded, start by running validation
    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch-1)

    for epoch in range(args.start_epoch, args.epochs):
        if args.rank==0 and distributed_sampler:
            train_sampler.set_epoch(epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            train(train_loader, r, optimizer, epoch, model_type, lengths, writer)

            # evaluate on validation set
            # prec1 = validate(val_loader, r, epoch)
            # if r.stage != r.num_stages: prec1 = 0

            # # remember best prec@1 and save checkpoint
            # best_prec1 = max(prec1, best_prec1)

            # should_save_checkpoint = args.checkpoint_dir_not_nfs or r.rank_in_stage == 0
            # if args.checkpoint_dir and should_save_checkpoint:
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'arch': args.arch,
            #         'state_dict': r.state_dict(),
            #         'best_prec1': best_prec1,
            #         'optimizer' : optimizer.state_dict(),
            #     }, args.checkpoint_dir, r.stage)


def train(train_loader, r, optimizer, epoch, model_type, lengths, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_type == runtime.IMAGE_CLASSIFICATION:
        top1 = AverageMeter()
        top5 = AverageMeter()
    temp_batch_time = AverageMeter()

    # switch to train mode
    n = r.num_iterations(loader_size=lengths[0])
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.train(n)
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)

    print(f"rank {rank} warmup minibatches = ", num_warmup_minibatches)

    # start num_warmup_minibatches forward passes
    for i in range(num_warmup_minibatches):
        r.run_forward()

    print("Warmup done...")
    for i in range(n - num_warmup_minibatches):
        # perform forward pass
        # print(f"Rank {rank} iteration {i+num_warmup_minibatches} / {n}")
        r.run_forward()
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.epochs, r, args.lr_policy, i, n)

        if is_last_stage():
            # measure accuracy and record loss
            output, target, loss = r.output, r.target, r.loss
            if model_type == runtime.IMAGE_CLASSIFICATION:
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                top1.update(prec1[0], output.size(0))
                top5.update(prec5[0], output.size(0))

            losses.update(loss.item(), output.size(0))
            batch_time.update(time.time() - end)
            temp_batch_time.update(time.time() - end)

            tput = r.num_ranks_in_stage*(1/batch_time.val)
            if writer and (r.rank_in_stage==0):
                n_iter = i
                writer.add_scalar(f'Loss/train', loss.item(), n_iter)
                writer.add_scalar(f'Perf/avg_throughput', tput, n_iter)

            # measure elapsed time
            end = time.time()
            epoch_time = (end - epoch_start_time) / 3600.0
            full_epoch_time = (epoch_time / (float(i+1)+num_warmup_minibatches)) * float(n) #changed

            if i % args.print_freq == 0:
                throughput = r.num_ranks_in_stage*(1/temp_batch_time.avg)
                if model_type == runtime.IMAGE_CLASSIFICATION:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Throughput: {throughput:.3f}\t'
                        'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                        'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, n, batch_time=batch_time, throughput=throughput,
                        epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                        loss=losses, top1=top1, top5=top5,
                        memory=(float(torch.cuda.memory_allocated()) / 10**9),
                        cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Throughput: {throughput:.3f}\t'
                        'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                        'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Perplexity: {latest_perp:.3f} ({avg_perp:.3f})'.format(
                        epoch, i, n, batch_time=batch_time,
                        epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                        loss=losses, latest_perp = np.exp(losses.val), avg_perp = np.exp(losses.avg), throughput=throughput,
                        memory=(float(torch.cuda.memory_allocated()) / 10**9),
                        cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                temp_batch_time.reset()
                import sys; sys.stdout.flush()
        else:
            pass
            # if i % args.print_freq == 0:
            #     print('Epoch: [{0}][{1}/{2}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
            #            epoch, i, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
            #            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
            #     import sys; sys.stdout.flush()

        # perform backward pass
        if args.fp16:
            r.zero_grad()
        else:
            optimizer.zero_grad()
        
        
        optimizer.load_old_params()
        try:
            r.run_backward()
        except Exception as e:
            print_msg(args.rank, "Failed backward pass of batch {}".format(i)+"\n"+str(e))
            exit()
        optimizer.load_new_params()
        optimizer.step()

    # finish remaining backward passes
    for i in range(num_warmup_minibatches):
        optimizer.zero_grad()
        optimizer.load_old_params()
        r.run_backward()
        optimizer.load_new_params()
        optimizer.step()

    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


def validate(val_loader, r, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.eval(n)
    if not is_first_stage(): val_loader = None
    r.set_loader(val_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running validation for %d minibatches" % n)

    with torch.no_grad():
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n - num_warmup_minibatches):
            # perform forward pass
            r.run_forward()
            r.run_ack()

            if is_last_stage():
                output, target, loss = r.output, r.target, r.loss

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), output.size(0))
                top1.update(prec1[0], output.size(0))
                top5.update(prec5[0], output.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                           epoch, i, n, batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5,
                           memory=(float(torch.cuda.memory_allocated()) / 10**9),
                           cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()

        if is_last_stage():
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        for i in range(num_warmup_minibatches):
             r.run_ack()

        # wait for all helper threads to complete
        r.wait()

        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


def save_checkpoint(state, checkpoint_dir, stage):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint.%d.pth.tar" % stage)
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, total_epochs, r, lr_policy, step, epoch_length):
    """ Adjusts learning rate based on stage, epoch, and policy.

    Gets learning rate for stage from runtime and adjusts based on policy.

    Supported LR policies:
         - step
         - polynomial decay
         - exponential decay
    """
    stage_base_lr = r.get_adjusted_learning_rate(base_lr=args.lr)

    if args.lr_warmup and epoch < 5:
        lr = stage_base_lr * float(1 + step + epoch*epoch_length)/(5.*epoch_length)

    else:
        if lr_policy == "step":
            lr = stage_base_lr * (0.1 ** (epoch // 30))
        elif lr_policy == "polynomial":
            power = 2.0
            lr = stage_base_lr * ((1.0 - (float(epoch) / float(total_epochs))) ** power)
        elif lr_policy == "exponential_decay":
            decay_rate = 0.97
            lr = stage_base_lr * (decay_rate ** (float(epoch) / float(total_epochs)))
        else:
            raise NotImplementedError

    if step % 100 == 0:
        print("Epoch: %d Step %d \tLearning rate: %f" % (epoch, step, lr))

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    optimizer.base_optimizer.update_lr(lr) #changed

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
