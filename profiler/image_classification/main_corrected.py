# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys; sys.path = [".."] + sys.path
import torchmodules.torchgraph as torchgraph
import torchmodules.torchlogger as torchlogger
import torchmodules.torchprofiler as torchprofiler
import torchmodules.torchsummary as torchsummary

import argparse
from collections import OrderedDict
import os
import shutil
import time

import torch
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
import torchvision.models as models

import models

# import models.densenet as densenet
# import models.inception as inception
# import models.mobilenet as mobilenet
# import models.nasnet as nasnet
# import models.resnext as resnext
# import models.squeezenet as squeezenet
# import models.vgg as vgg
# import models.alexnet as alexnet
# import models.lenet as lenet

import gc
print(models.__file__)

def print_gpu_tensors():
    print("\n~~~~~~~~~~~\n")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
    print("\n~~~~~~~~~~~\n")

def is_parameterless(layer):
    return not list(layer.parameters())

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
# model_names = sorted(name for name in mobilenet.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(mobilenet.__dict__[name]))
# model_names += sorted(name for name in nasnet.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(nasnet.__dict__[name]))
# model_names += sorted(name for name in resnext.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(resnext.__dict__[name]))
# model_names += sorted(name for name in vgg.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(vgg.__dict__[name]))
# model_names += sorted(name for name in alexnet.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(alexnet.__dict__[name]))
# model_names += sorted(name for name in lenet.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(lenet.__dict__[name]))

# print(model_names)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
                    # choices=model_names,
                    # help='model architecture: ' +
                    #     ' | '.join(model_names) +
                    #     ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--log_activations', action='store_true',
                    help="Log activations")
parser.add_argument('--log_activations_freq', default=5, type=int,
                    help="Frequency at which activations and gradients should be logged")
parser.add_argument('--log_activations_directory', default="activations",
                    help="Activations directory")
parser.add_argument('--profile_directory', default="profiles/",
                    help="Profile directory")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--log_reduce_times', action='store_true',
                    help="Log reduce times")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    default=False, help="Use synthetic data (default: false)")
parser.add_argument('-v', '--verbose', action='store_true',
                    help="Controls verbosity while profiling")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--dataset-name', default='ImageNet', type=str,
                    help='distributed backend')

best_prec1 = 0


def create_graph(model, train_loader, summary, directory):
    """Given a model, creates and visualizes the computation DAG
       of the model in the passed-in directory."""
    graph_creator = torchgraph.GraphCreator(model, summary, module_whitelist=['CombinedEmbedding', 'TransformerEncoderLayerWithMask', 'FinalFCLayer'])
    graph_creator.hook_modules(model)
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(input)
        if i >= 0:
            break
    graph_creator.unhook_modules()
    graph_creator.persist_graph(directory)

def is_parameterless(layer):
    return not list(layer.parameters())

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

class ImageNet_util(object):
    def __init__(self):
        pass 

    def get_model(self, arch):
        if args.arch.startswith('densenet'):
            import models.ImageNet.densenet as densenet
            model = densenet.__dict__[arch]()
        elif args.arch.startswith('inception_v3'):
            model = ImageNet.inception.__dict__[arch]()
        elif args.arch.startswith('mobilenet'):
            model = ImageNet.mobilenet.__dict__[arch]()
        elif args.arch.startswith('nasnet'):
            model = ImageNet.nasnet.__dict__[arch]()
        elif args.arch.startswith('resnext'):
            import models.ImageNet.resnext as resnext
            model = resnext.__dict__[arch]()
        elif args.arch.startswith('squeezenet'):
            model = ImageNet.squeezenet.__dict__[arch]()
        elif args.arch.startswith('vgg'):
            import models.ImageNet.vgg as vgg
            model = vgg.__dict__[arch]()
        elif args.arch.startswith('alexnet'):
            import models.ImageNet.alexnet as alexnet
            model = alexnet.__dict__[arch]()
        elif args.arch.startswith('lenet'):
            model = ImageNet.lenet.__dict__[arch]()
        else:
            print("Architecture in Imagenet not found.. aborting")
            exit(-1)
        return model
    
    def get_dataset(self, arch, data_dir=None):
        if not data_dir:
            if arch == 'inception_v3':
                train_dataset = SyntheticDatasetImageClassification((3, 299, 299), 100)
            else:
                train_dataset = SyntheticDatasetImageClassification((3, 224, 224), 100)
        else:
            traindir = os.path.join(data_dir, 'train')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if arch == 'inception_v3':                
                train_dataset = datasets.ImageFolder(traindir, 
                                                transform=transforms.Compose([
                                                    transforms.RandomResizedCrop(299),
                                                    transforms.ToTensor(),
                                                    normalize 
                                                ]))
            else:
                train_dataset = datasets.ImageFolder(traindir,
                                                transform=transforms.Compose([
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize
                                                ]))
        return train_dataset

class CIFAR10_util(object):
    def __init__(self):
        pass 

    def get_model(self, arch):
        if args.arch.startswith('vgg'):
            import models.CIFAR_10.vgg as vgg
            model = vgg.__dict__[arch]()
        else:
            print("Architecture in Imagenet not found.. aborting")
            exit(-1)
        return model
    
    def get_dataset(self, arch, data_dir=None):
        if not data_dir:
            train_dataset = SyntheticDatasetImageClassification((3, 32, 32), 100)
        else:
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform =transform)
        return train_dataset

class LM_util(object):
    def __init__(self):
        pass 

    def get_model(self, arch):
        import models.language_modelling.transformer as transformer
        model = transformer.__dict__[arch]()
        return model 

    def get_dataset(self, arch, data_dir=None):
        train_dataset = SyntheticDatasetLanguageModelling(50257, 256, 100)
        return train_dataset

def profile_train(train_loader, model, criterion, optimizer):
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    NUM_STEPS_TO_PROFILE = 100 # profile 100 steps or minibatches

    # switch to train mode
    model.train()

    layer_timestamps = []
    data_times = []

    iteration_timestamps = []
    opt_step_timestamps = []
    data_timestamps = []
    
    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_pid = os.getpid()
        data_time = time.time() - start_time
        data_time_meter.update(data_time)
        with torchprofiler.Profiling(model, module_whitelist=[]) as p:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            input.requires_grad=True
            # compute output
            output = model(input)
            if isinstance(output, tuple):
                loss = sum((criterion(output_elem, target) for output_elem in output))
            else:
                loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            try:
                loss.backward()
            except RuntimeError:
                pass
            optimizer_step_start = time.time()
            optimizer.step()

            end_time = time.time()
            iteration_time = end_time - start_time
            batch_time_meter.update(iteration_time)

            if i >= NUM_STEPS_TO_PROFILE:
                break
        p_str = str(p)
        layer_timestamps.append(p.processed_times())
        data_times.append(data_time)

        if args.verbose:
            print('End-to-end time: {batch_time.val:.3f} s ({batch_time.avg:.3f} s)'.format(
                  batch_time=batch_time_meter))

        iteration_timestamps.append({"start": start_time * 1000 * 1000,
                                     "duration": iteration_time * 1000 * 1000})
        opt_step_timestamps.append({"start": optimizer_step_start * 1000 * 1000,
                                    "duration": (end_time - optimizer_step_start) * 1000 * 1000, "pid": os.getpid()})
        data_timestamps.append({"start":  start_time * 1000 * 1000,
                                "duration": data_time * 1000 * 1000, "pid": data_pid})
        
        start_time = time.time()

    layer_times = []
    tot_accounted_time = 0.0
    if args.verbose:
        print("\n==========================================================")
        print("Layer Type    Forward Time (ms)    Backward Time (ms)")
        print("==========================================================")

    for i in range(len(layer_timestamps[0])):
        layer_type = str(layer_timestamps[0][i][0])
        layer_forward_time_sum = 0.0
        layer_backward_time_sum = 0.0
        for j in range(len(layer_timestamps)):
            layer_forward_time_sum += (layer_timestamps[j][i][2] / 1000)
            layer_backward_time_sum += (layer_timestamps[j][i][5] / 1000)
        layer_times.append((layer_type, layer_forward_time_sum / len(layer_timestamps),
                                    layer_backward_time_sum / len(layer_timestamps)))
        if args.verbose:
            print(layer_times[-1][0], layer_times[-1][1], layer_times[-1][2])
        tot_accounted_time += (layer_times[-1][1] + layer_times[-1][2])

    print()
    print("Total accounted time: %.3f ms" % tot_accounted_time)
    return layer_times, (sum(data_times) * 1000.0) / len(data_times)

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.profile = True

    print("=> creating model '{}'".format(args.arch))

    if args.dataset_name == "ImageNet":
        dataset_util = ImageNet_util()
    elif args.dataset_name == "CIFAR10":
        dataset_util = CIFAR10_util()
    elif args.dataset_name in ["wikitext-2", "wikitext-103"]:
        dataset_util = LM_util()


    model = dataset_util.get_model(args.arch)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    cudnn.benchmark = True
  
    if args.synthetic_data:
        train_dataset = dataset_util.get_dataset(args.arch)
    else:
        train_dataset = dataset_util.get_dataset(args.arch, args.data_dir)
  

    print("Memory in bytes - " + str(torch.cuda.memory_allocated() ))
    
    # model = model.cpu()
    # print("Memory in bytes - " + str(torch.cuda.memory_allocated() ))
    # exit()
    train_sampler = None

    print("Batch size = ",args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, sampler=train_sampler)


    print("Collecting profile...")
    for i, (model_input, _) in enumerate(train_loader):
        model_input = model_input.cuda()
        ip_size = model_input.size()
        if i >= 0:
            break
    summary = torchsummary.summary(model=model, module_whitelist=['CombinedEmbedding', 'TransformerEncoderLayerWithMask', 'FinalFCLayer'], model_input=(model_input,), verbose=args.verbose, device="cuda")
    
    del model_input
    # print(summary)
    # model = model.cpu()
    print("Memory in bytes - " + str(torch.cuda.memory_allocated() ))
    
    
    prof_num = 100
    print("Profiling Data Loader..")
    
    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        print("Iteration ",i, end="\r")
        input.cuda()
        target.cuda()
        #torch.cuda.synchronize()
        if i == prof_num:
            break 
    end_time = time.time() 
    data_time = (end_time - start_time)*1000/prof_num
    print("Data loading time..", data_time, "ms")

    prof_num = 150

    output, target = next(iter(train_loader)) 
    print("Collecting profile...")
    for i, (model_input, _) in enumerate(train_loader):
        model_input = model_input.cuda()
        if i >= 0:
            break


    summary = torchsummary.summary(model=model, module_whitelist=[], model_input=(model_input,),
                                    verbose=args.verbose, device="cuda")
    per_layer_times, data_time = profile_train(train_loader, model, criterion, optimizer)
    summary_i = 0
    per_layer_times_i = 0
    while summary_i < len(summary) and per_layer_times_i < len(per_layer_times):
        summary_elem = summary[summary_i]
        per_layer_time = per_layer_times[per_layer_times_i]
        if str(summary_elem['layer_name']) != str(per_layer_time[0]):
            summary_elem['forward_time'] = 0.0
            summary_elem['backward_time'] = 0.0
            summary_i += 1
            continue
        summary_elem['forward_time'] = per_layer_time[1]
        summary_elem['backward_time'] = per_layer_time[2]
        summary_i += 1
        per_layer_times_i += 1
    summary.append(OrderedDict())
    summary[-1]['layer_name'] = 'Input'
    summary[-1]['forward_time'] = data_time
    summary[-1]['backward_time'] = 0.0
    summary[-1]['nb_params'] = 0.0
    summary[-1]['output_shape'] = [args.batch_size] + list(model_input.size()[1:])
    create_graph(model, train_loader, summary,
                    os.path.join(args.profile_directory, args.arch))
    print("...done!")
    return
    
    for layer_details in summary:
        layer = layer_details['layer_obj']
        input_ = output.detach() 
        grad_ = torch.randn(*layer_details['output_shape'])
        layer.cuda()
        
        for param in layer.parameters():
            param.requires_grad = True

        input_.requires_grad = is_parameterless(layer)
        input_ = input_.cuda()
        grad_ = grad_.cuda() 
        
        fw_time = 0
        bw_time = 0 
        print("Profiling ",layer_details['layer_name'])
        
        for _ in range(prof_num):
            #FW pass 
            start = time.time()
            output = layer(input_)
            torch.cuda.synchronize()
            end = time.time()
            fw_time += (end-start)
            tmp = end-start
            start = time.time()
            output.backward(grad_)
            torch.cuda.synchronize()
            end = time.time() 
            bw_time += (end-start)
        # print("Memory in bytes (before deletion) - " + str(torch.cuda.memory_allocated() ))
        layer.cpu()
        # print("Memory in bytes (after deletion) - " + str(torch.cuda.memory_allocated() ))
        del input_
        del grad_
        print("FW time {} ms BW time {} ms".format(fw_time * 1000/prof_num, bw_time * 1000/prof_num))
        layer_details['forward_time'] = fw_time * 1000 /prof_num
        layer_details['backward_time'] = bw_time * 1000 /prof_num

    # per_layer_times, data_time = profile_train(train_loader, model, criterion, optimizer, summary)
    # summary_i = 0
    # per_layer_times_i = 0
    # while summary_i < len(summary) and per_layer_times_i < len(per_layer_times):
    #     summary_elem = summary[summary_i]
    #     per_layer_time = per_layer_times[per_layer_times_i]
    #     if str(summary_elem['layer_name']) != str(per_layer_time[0]):
    #         summary_elem['forward_time'] = 0.0
    #         summary_elem['backward_time'] = 0.0
    #         summary_i += 1
    #         continue
    #     summary_elem['forward_time'] = per_layer_time[1]
    #     summary_elem['backward_time'] = per_layer_time[2]
    #     summary_i += 1
    #     per_layer_times_i += 1
    summary.append(OrderedDict())
    summary[-1]['layer_name'] = 'Input0'
    summary[-1]['forward_time'] = data_time
    summary[-1]['backward_time'] = 0.0
    summary[-1]['nb_params'] = 0.0
    summary[-1]['output_shape'] = [args.batch_size] + list(ip_size[1:])
    model.cuda()
    create_graph(model, train_loader, summary,
                    os.path.join(args.profile_directory, args.arch))
    print("...done!")
    return




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

if __name__ == '__main__':
    main()
