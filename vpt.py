from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging
import pdb

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model[0].parameters(), model[0].parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class VPT_wrapper(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, args,model, prompter,steps=1, mt_alpha=0.99):
        super().__init__()

        self.steps = steps
        self.N = args.N
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
#`model_state`: 字典，包含了模型的所有参数的状态。这个状态可以用来重置模型的参数，以便在适应过程中进行恢复。
#`optimizer_state`: 字典，包含了优化器的状态。这个状态可以用来重置优化器的状态，以便在适应过程中进行恢复。
#`ema_model`: 模型的深拷贝，用于指数移动平均。在适应过程中，模型的参数会被更新，但是指数移动平均需要使用模型的旧参数。因此，我们需要一个深拷贝来保存模型的旧参数。
#`model_anchor`: 模型的深拷贝，用于重置模型的参数。在适应过程中，模型的参数会被更新，但是在某些情况下，我们需要重置模型的参数。因此，我们需要一个深拷贝来保存模型的初始参数。
        
        self.stu_prompter = deepcopy(prompter)
        self.optimizer = torch.optim.SGD(self.stu_prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        self.model = nn.Sequential(self.stu_prompter,model)
        self.prompter_state,self.ema_prompter,self.optimizer_state =\
            copy_model_and_optimizer(prompter, self.optimizer)
        # self.prompter_state,self.model, self.model_ema,self.optimizer_state  = \

        self.model_ema = nn.Sequential(self.ema_prompter,model)

        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        # self.rst = rst_m
        # self.ap = ap

    def forward(self, x):
        # if self.episodic:
        #     self.reset()
        if self.model[0].training:
            for _ in range(self.steps):
                outputs_ema,outputs,_,_ = self.forward_and_adapt(x, self.optimizer)
            return outputs_ema,outputs
        else:
            outputs=self.model(x)
            return outputs
            

        

    def reset(self):
        if self.prompter_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.prompter_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.prompter_state, self.model_ema[0],self.optimizer_state  = \
            copy_model_and_optimizer(self.model[0], self.optimizer)



    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x,optimizer):
        outputs=self.model(x)
        # Teacher Prediction
        # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = self.N
        outputs_emas = []
        for i in range(N):
            outputs_  = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        # if anchor_prob.mean(0)<self.ap: #预测不自信，用增强平均值增加模型鲁棒性，减少模型对输入数据的依赖，提高泛化能力
            #`outputs_emas` 是一个包含了 `N` 个张量的列表，每个张量的维数都是 `(batch_size, num_classes)`
            # `torch.stack(outputs_emas)` 将这 `N` 个张量沿着新的维度进行堆叠，得到一个新的张量，其维数为 `(N, batch_size, num_classes)`。
            # `.mean(0)` 计算沿着第一个维度的平均值，得到一个维数为 `(batch_size, num_classes)` 的张量，即 `outputs_ema`。
            # outputs_ema = torch.stack(outputs_emas).mean(0)
        # else: #自信，表明数据变化程度较小？使用模型输出的指数移动平均值作为模型的输出可以减少模型的震荡，从而提高模型的稳定性。
            #因为指数移动平均值是对模型输出的历史值进行加权平均得到的，可以减少模型输出的波动，从而提高模型的稳定性。
        # outputs_ema = standard_ema
        # Student update
        outputs_ema = torch.stack(outputs_emas).mean(0)
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        l2_norm = 0
        max_l1_norm = 0
        for param in self.model[0].parameters():
            if param.grad is not None:
                l2_norm += torch.norm(param.grad, p=2)
                max_l1_norm = max(max_l1_norm, torch.norm(param.grad, p=1))
        # print("L2 norm of gradients:", l2_norm.item())
        # print("Max L1 norm of gradients:", max_l1_norm.item())

        # pdb.set_trace()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update ;self.model:当前模型参数，self.model_ema:指数移动平均模型参数
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restorepro
        # if True:
        #     for nm, m  in self.model.named_modules():
        #         for npp, p in m.named_parameters():
        #             if npp in ['weight', 'bias'] and p.requires_grad:
        #                 mask = (torch.rand(p.shape)<self.rst).float().cuda() 
        #                 with torch.no_grad():
        #                     p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema,outputs,l2_norm,max_l1_norm


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(prompter, optimizer):
    """Copy the prompter and optimizer states for resetting after adaptation."""
    prompter_state = deepcopy(prompter.state_dict())
    # prompter_anchor = deepcopy(prompter)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_prompter = deepcopy(prompter)
    for param in ema_prompter.parameters():
        param.detach_() #detach： 防止计算指数移动平均时，梯度回传到模型参数
    return prompter_state,ema_prompter,optimizer_state


def load_model_and_optimizer(model, optimizer, prompter_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.prompter.load_state_dict(prompter_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model,prompter):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    prompter.train()
    # disable grad, to (re-)enable only what we update
    model.eval()
    model.requires_grad_(False)
    # enable all trainable
    for param in prompter.parameters():
        param.requires_grad_(True)
    for param in model.parameters():
        param.requires_grad_(False)
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.requires_grad_(True)
    #         # force use of batch stats in train and eval modes
    #         m.track_running_stats = False
    #         m.running_mean = None
    #         m.running_var = None
    #     else:
    #         m.requires_grad_(True)
    return nn.Sequential(prompter,model)


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
