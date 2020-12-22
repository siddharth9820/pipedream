# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os
from torch.autograd import Variable
from torch.autograd import Function

import time


def is_parameterless(layer):
    return not list(layer.parameters())

def make_pre_hook_bw(module_obj, profiler_obj):
    def hook(grad):
        profiler_obj.record['backward'].append(
            (module_obj, time.time(), os.getpid()))
    return hook

def none_grad(layer):
    for name, param in layer.named_parameters():
        if param.grad != None:
            del param.grad 
            param.grad = None


class Profiling(object):
    def __init__(self, model, module_whitelist):
        if isinstance(model, torch.nn.Module) is False:
            raise Exception("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.module_whitelist = module_whitelist
        self.record = {'forward':[], 'backward': []}
        self.profiling_on = True
        self.forward_original_methods = {}
        self.hook_done = False
        self.unhook_done = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        tot_time = 0.0

        ret = ""
        ret += "\n===============================================================\n"
        ret += "Forward Times\n"
        ret += "===============================================================\n"
        for i in range(len(self.record['forward'])):
            record_item = self.record['forward'][i]
            ret += "layer{:3d}:\t{:.6f} ms\t({})\n".format(
                i + 1, (record_item[3] - record_item[1]) * 1000, record_item[0])
            tot_time += ((record_item[3] - record_item[1]) * 1000)

        ret += "\n===============================================================\n"
        ret += "Backward Times\n"
        ret += "===============================================================\n"
        self.record['backward'].reverse()
        for i in range(len(self.record['forward'])):
            try:
                record_item = self.record['backward'][i]
                ret += "layer{:3d}:\t{:.6f} ms\t({})\n".format(
                    i + 1, (record_item[3] - record_item[1]) * 1000, record_item[0])
                tot_time += ((record_item[3] - record_item[1]) * 1000)
            except Exception as e:
                # Oops, this layer doesn't have metadata as needed.
                pass

        ret += ("\nTotal accounted time in forward and backward pass: %.6f ms" % tot_time)
        return ret

    def processed_times(self):
        processed_times = []
        forward_i = 0
        backward_i = 0
        last_forward_i = -1
        last_backward_i = -1
        while forward_i < len(self.record['forward']) and backward_i < len(self.record['backward']):
            forward_record_item = self.record['forward'][forward_i]
            backward_record_item = self.record['backward'][backward_i]
            if forward_record_item[0] != backward_record_item[0]:
                forward_i += 1
                continue
            if forward_i != (last_forward_i + 1):
                forward_i = last_forward_i
                last_backward_i = backward_i
                backward_i += 1
            else:
                last_forward_i, last_backward_i = forward_i, backward_i
                forward_i += 1
                backward_i += 1
            forward_time = (forward_record_item[3] - forward_record_item[1])
            backward_time = (backward_record_item[3] - backward_record_item[1])
            processed_times.append((forward_record_item[0],
                                    forward_record_item[1] * 1000 * 1000,
                                    forward_time * 1000 * 1000, forward_record_item[2],
                                    backward_record_item[1] * 1000 * 1000,
                                    backward_time * 1000 * 1000, backward_record_item[2]))
        return processed_times

    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.hook_modules(self.model)
        self.profiling_on = True
        return self

    def stop(self):
        self.profiling_on = False
        if self.unhook_done is False:
            self.unhook_done = True
            self.unhook_modules(self.model)
        return self

    def hook_modules(self, module):
        this_profiler = self
        sub_modules = module.__dict__['_modules']

        for name, sub_module in sub_modules.items():

            # nn.Module is the only thing we care about.
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_module_name = sub_module.__class__.__name__
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0 and sub_module_name not in self.module_whitelist:
                #
                # Recursively visit this module's descendants.
                #
                self.hook_modules(sub_module)
            else:
                # sub_module.reset_hooks()
                #
                # Hook nn.Module with no descendants.
                #

                # Wrapper function to "forward", with timer to record how long
                # forward pass takes.
                def forward_wrapper(self, *input):
                    assert(len(input) == 1) , "not suuporting lists or tuples.."
                    name = self.__class__.__name__                       
                    
                    if is_parameterless(self) and name != "ReLU":
                        input[0].requires_grad = True 
                       
                    pid = os.getpid()
                    start_time = time.time()
                   
                    if name == "ReLU":
                        with torch.no_grad():
                            result = this_profiler.forward_original_methods[self](*input)
                    else:
                        result = this_profiler.forward_original_methods[self](
                            *input)

                    assert not isinstance(result, (list, tuple)), "not supporting lists and tuples.."
                    torch.cuda.synchronize()
                    stop_time = time.time()

                    if (this_profiler.profiling_on):
                        this_profiler.record['forward'].append((self, start_time, pid, stop_time))
                    print("Forward of {} - Time {} seconds Mem {} ".format(self, stop_time-start_time, torch.cuda.memory_allocated()/(1000000000)))
                    tot = stop_time - start_time
                    
                    if name != "ReLU":
                        grad = torch.zeros_like(result).cuda()
                        torch.cuda.synchronize()
                        start_time = time.time()
                        result.backward(grad)
                        torch.cuda.synchronize()
                        stop_time = time.time()
                        del grad
                    else:
                        start_time = time.time()
                        stop_time = start_time + tot 
                    
                    if (this_profiler.profiling_on):
                        this_profiler.record['backward'].append(
                                (self, start_time, pid, stop_time))
                    print("Backward of {} - Time {} seconds Mem {}".format(self,
                                                                   stop_time-start_time, torch.cuda.memory_allocated()/(1000000000)))
                    # result.register_hook(make_pre_hook_bw(self, this_profiler))
                    # none_grad(self)
                    del input
                                       
                    return result.detach()

                # Replace "forward" with "forward_wrapper".
                if sub_module not in this_profiler.forward_original_methods:
                    this_profiler.forward_original_methods.update({sub_module:
                                                                   sub_module.forward})
                    sub_module.forward = forward_wrapper.__get__(sub_module, sub_module.__class__)

                # Start timer for backward pass in pre_hook; then stop timer
                # for backward pass in post_hook.
                # def backward_pre_hook(*args):
                #     if (this_profiler.profiling_on):
                #         this_profiler.record['backward'].append((args[0], time.time(), os.getpid()))

                # def backward_post_hook(*args):
                #     idx = -1
                #     if not this_profiler.profiling_on:
                #         return
                #     while args[0] != this_profiler.record['backward'][idx][0]:
                #         idx -= 1
                #         if (-idx) == len(this_profiler.record['backward']):
                #             return
                #     torch.cuda.synchronize()
                #     this_profiler.record['backward'][idx] = (this_profiler.record['backward'][idx][0],
                #                                              this_profiler.record['backward'][idx][1],
                #                                              this_profiler.record['backward'][idx][2],
                #                                              time.time())
                # # sub_module.register_backward_pre_hook(backward_pre_hook)
                # sub_module.register_backward_hook(backward_post_hook)

    def unhook_modules(self, module):
        sub_modules = module.__dict__['_modules']

        for name, sub_module in sub_modules.items():
            # nn.Module is the only thing we care about.
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_module_name = sub_module.__class__.__name__
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0 and sub_module_name not in self.module_whitelist:
                #
                # Recursively visit this module's descendants.
                #
                self.unhook_modules(sub_module)
            else:
                # sub_module.reset_hooks()
                sub_module.forward = self.forward_original_methods[sub_module]
