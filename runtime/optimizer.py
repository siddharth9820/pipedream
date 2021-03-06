# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.optim
import time
import copy

from collections import deque  # Efficient ring buffer implementation.

class Version:
    def __init__(self, version=0):
        self.version = version

    def __repr__(self):
        return "v%d" % self.version

    def incr(self):
        return Version(version=self.version+1)

class Adam_Base(object):
    """An Adam optimizer written for myelin which can handle weight stashing
    """    
    def __init__(self, parameters, learning_rate=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8, id=None):
        """Initialise the Adam optimizer
        Args:
            parameters (Iterator[torch.nn.parameter.Parameter]): parameter iterator of a module i.e. module.parameters()
            learning_rate (int): learning rate. Defaults to 0.001
            beta_1 (float) : decay factor for first moment. Defaults to 0.9 
            beta_2 (float) : decay factor for second moment. Defaults to 0.999
            id ((int,int), optional): The ID of the chare invoking this class. Defaults to None.
        """        
        self.parameters = list(parameters) 
        self.learning_rate = learning_rate 
        self.beta1 = beta_1
        self.beta2 = beta_2 
        self.t = 1
        self.eps = eps
        
        self.m = []
        self.v = []
        for param in self.parameters:
            self.m.append(torch.zeros_like(param.data).cuda())
            self.v.append(torch.zeros_like(param.data).cuda())

    def step(self, params):
        """Update the weights of the module using the gradient data
        Args:
             params (Iterator[torch.nn.parameter.Parameter]): parameter iterator of a module i.e. module.parameters()
        """ 
        t = self.t
        beta1, beta2 = self.beta1, self.beta2
        alpha = self.learning_rate * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        eps = self.eps        
        for i,param in enumerate(params):
            if param.grad is not None:
                m = self.m[i]
                v = self.v[i]
                g = param.grad
                m.mul_(beta1)
                m.add_(g, alpha=1-beta1)
                v.mul_(beta2)
                v.add_(g**2, alpha=1-beta2)
                param.data.add_(m/(torch.sqrt(v) + eps), alpha=-alpha)
        self.t+=1 
    
    def update_lr(self, new_lr):
        """Update the learning rate of the optimizer
        Args:
            new_lr (float): new learning rate
        """        
        self.learning_rate = new_lr
    
    def zero_grad(self):
        """Zero out all parameter gradients
        """        
        for param in self.parameters:
            if param.grad != None:
                param.grad.zero_()

class SGD_Base(object):
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters 
        self.learning_rate = learning_rate 
    
    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.learning_rate * param.grad.data

    def update_lr(self, new_lr):
        self.learning_rate = new_lr
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad != None:
                param.grad.zero_()

    def average_grad(self, update_interval):
        for param in self.parameters:
            if param.grad != None:
                param.grad /= update_interval
        



class OptimizerWithWeightStashing(torch.optim.Optimizer):
    """Wrapper class that adds weight stashing to a vanilla torch.optim.Optimizer.

    Arguments:
        - optim_name: the name of optimizer, required to create the corresponding
                      base_optimizer (torch.optim.{optim_name}).
        - optimizer_args: the keyword arguments passed to base_optimizer.
    """

    def __init__(self, optim_name, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, verbose_freq=0, macrobatch=False, type='SGD', **optimizer_args):
        self.modules = modules
        self.master_parameters = master_parameters
        self.model_parameters = model_parameters  # model_parameters is None if not fp16.
        self.loss_scale = loss_scale
        self.lr = 1e-3

        # Only need at most 2 versions if using macrobatching.
        if macrobatch:
            num_versions = min(2, num_versions)
        self.num_versions = num_versions
        # self.base_optimizer = getattr(torch.optim, optim_name)(
        #     master_parameters, **optimizer_args)
        
        if type == "SGD":
            self.base_optimizer = SGD_Base(master_parameters, self.lr)
        elif type == "Adam":
            self.base_optimizer = Adam_Base(master_parameters, self.lr)
        else:
            raise NotImplementedError

        self.latest_version = Version()
        self.current_version = Version()
        self.initialize_queue()
        self.verbose_freq = verbose_freq
        self.batch_counter = 0

        # If macrobatching, push and pop versions at the right rate.
        if macrobatch:
            self.update_interval = self.num_versions
        else:
            self.update_interval = 1

    def __getattr__(self, key):
        """Relay the unknown key to base_optimizer."""
        return getattr(self.base_optimizer, key)

    def initialize_queue(self):
        self.queue = deque(maxlen=self.num_versions)
        for i in range(self.num_versions):
            self.queue.append(self.get_params(clone=True))
        self.buffered_state_dicts = self.queue[0][0]

    def get_params(self, clone):
        if clone:
            state_dicts = []
            for module in self.modules:
                state_dict = module.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].clone()
                state_dicts.append(state_dict)
        else:
            for i, module in enumerate(self.modules):
                state_dict = module.state_dict()
                for key in state_dict:
                    # Running_mean and running_var for batchnorm layers should
                    # accumulate normally.
                    if "running_" in key:
                        continue
                    if "mask" in key:
                        self.buffered_state_dicts[i][key] = state_dict[key].clone()
                    else:
                        self.buffered_state_dicts[i][key].copy_(state_dict[key])
            state_dicts = self.buffered_state_dicts
        return state_dicts, self.latest_version

    def set_params(self, state_dicts, version):
        for (state_dict, module) in zip(state_dicts, self.modules):
            cur_state_dict = module.state_dict()
            for key in state_dict:
                # Don't update running_mean and running_var; these should
                # accumulate normally.
                # mask might have a different shape, so don't copy it to
                # the module this way.
                if "running_" in key or "mask" in key: #added
                    state_dict[key] = cur_state_dict[key] #added
                cur_state_dict[key].data = state_dict[key].data #added
            # module.load_state_dict(state_dict) #comment out

            # Load the mask.
            for key in state_dict:
                if "mask" in key:
                    attribute_names = key.split(".")
                    attribute = module
                    for attribute_name in attribute_names:
                        attribute = getattr(attribute, attribute_name)
                    # NOTE: Do we need to clone here?
                    attribute = state_dict[key]
        self.current_version = version

    def load_old_params(self):
        if self.num_versions > 1:
            self.set_params(*self.queue[0])

    def load_new_params(self):
        if self.num_versions > 1:
            self.set_params(*self.queue[-1])

    def zero_grad(self):
        if self.batch_counter % self.update_interval == 0:
            self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        # Update the gradient every `update_interval` steps.
        if self.batch_counter % self.update_interval != self.update_interval - 1:
            self.batch_counter += 1
            return None

        log_timing = self.verbose_freq > 0 and self.batch_counter % self.verbose_freq == 0
        if log_timing:
            start_time = time.time()
        if self.model_parameters is not None:
            import apex.fp16_utils as fp16_utils
            fp16_utils.model_grads_to_master_grads(self.model_parameters,
                                                   self.master_parameters)
            # TODO: This division might not be in the right place, given that
            # scaling happens right after. Look into this if problems arise.
            if self.loss_scale != 1.0:
                for parameter in self.master_parameters:
                    parameter.grad.data = parameter.grad.data / self.loss_scale

        # for p in self.param_groups[0]['params']:
        #     if p.grad is not None:
        #         p.grad.div_(self.update_interval)

        # loss = self.base_optimizer.step()

        self.base_optimizer.average_grad(self.update_interval)

        self.base_optimizer.step()

        if self.model_parameters is not None:
            import apex.fp16_utils as fp16_utils
            fp16_utils.master_params_to_model_params(self.model_parameters,
                                                     self.master_parameters)
        self.latest_version = self.latest_version.incr()
        if self.num_versions > 1:
            self.buffered_state_dicts = self.queue[0][0]
            self.queue.append(self.get_params(clone=False))

        if log_timing:
            print("Optimizer step took: %.3f" % (time.time() - start_time))
        self.batch_counter += 1
        
        # return loss


