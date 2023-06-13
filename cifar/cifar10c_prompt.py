from __future__ import print_function
import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import norm
import cotta
import vpt

import logging

import argparse
import os
from tqdm import tqdm
import time
import random
import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

# import clip
import prompters
from visualprompting.utils import AverageMeter, ProgressMeter, save_checkpoint
from visualprompting.utils import cosine_lr, convert_models_to_fp32, refine_classname

from conf2 import cfg, load_cfg_fom_args
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy
from robustbench.model_zoo.enums import ThreatModel
import torch.optim as optim
import pandas as pd
import sys
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from vpt import VPT_wrapper,get_tta_transforms
from my_utils import Logger
import math
from tensorboardX import SummaryWriter 
import json


logger = logging.getLogger(__name__)

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=0,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    # parser.add_argument('--model', type=str, default='clip')
    # parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='fixed_loc',
                        choices=['fixed_loc','random_loc','padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=str, default='2_5',
                        help='size for visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='/home/cll/work/data/classification/CIFAR100',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=32,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')
    parser.add_argument('--N', type=int, default=4,
                        help='number of augmentations')
    parser.add_argument('--cfg', dest='cfg_file',default='cfgs/cifar10/vpt.yaml',
                        help='optional config file',)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    parser.add_argument('--mt_alpha', type=float, default=0.99,
                        help='coefficient for ema')
    parser.add_argument('--norm', default=False,
                        action="store_true",
                        help='add norm adaptation')
    args = parser.parse_args()


    return args


device = "cuda" if torch.cuda.is_available() else "cpu"


args = parse_option()
    # args.model_folder = os.path.join('outputs', args.filename)
    # if not os.path.isdir(args.model_folder):
    #     os.makedirs(args.model_folder)

    
description='"CIFAR-10-C evaluation.'
load_cfg_fom_args(args,description)
args.model_folder = args.save_path
tta_log_nm='/tta_prompt_'+cfg.LOG_DEST.replace('.txt','.log')
sys.stdout = Logger(args.model_folder+tta_log_nm, sys.stdout)
args.log_path=args.save_path.replace('output','logs2')

def evaluate(description):
    # load_cfg_fom_args(args,description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    if cfg.MODEL.ADAPTATION == "vpt":
        logger.info("test-time adaptation: VPT")
        prompter = prompters.__dict__[args.method](args).to(device)
        ckpt = torch.load(args.resume)
        prompter.load_state_dict(ckpt['state_dict'])
        model = setup_vpt(base_model,prompter)
        model.optimizer.load_state_dict(ckpt['optimizer'])
    # evaluate on each severity and type of corruption in turn
    
    df0 = pd.DataFrame(data=[vars(args).values()], columns=vars(args).keys())
    df_all=pd.DataFrame()

    prev_ct = "x0"
    for severity in cfg.CORRUPTION.SEVERITY:
        error_sum_stu=0
        error_sum_ema = 0
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # continual adaptation for all corruption 
            df_=df0.copy(deep=True)

            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc_stu,acc_ema = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            err_stu = 1. - acc_stu
            err_ema = 1. - acc_ema
            error_sum_stu += err_stu
            error_sum_ema += err_ema
            logger.info(f"error % [{corruption_type}{severity}]: student {err_stu:.2%}, ema {err_ema:.2%}")
            df_['corruption type']=corruption_type
            df_['severity']=severity
            df_['adaptation error stu']=err_stu
            df_['adaptation error ema']=err_ema
            df_all = df_all.append(df_, ignore_index=True)
            
        df_=df0.copy(deep=True)
        err_mean_stu=error_sum_stu/len(cfg.CORRUPTION.TYPE)
        err_mean_ema = error_sum_ema/len(cfg.CORRUPTION.TYPE)
        df_['corruption type']='mean'
        df_['severity']=severity
        df_['adaptation error stu']=err_mean_stu
        df_['adaptation error ema']=err_mean_ema
        df_all = df_all.append(df_, ignore_index=True)           
            
 
            # acc= accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            # err = 1. - acc
            # logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
    result_dir=args.save_path.replace('output','results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    df_all.to_csv(f'{result_dir}/'+cfg.LOG_DEST.replace('.txt','.csv'),index=False)

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_vpt(model,prompter):
    model,prompter = vpt.configure_model(args,model,prompter)
    vpt_model = vpt.VPT_wrapper(args,model,prompter)
    return vpt_model 

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

def accuracy(model: torch.nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc_stu = 0.
    acc_ema = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            outputs_ema,outputs_stu = model(x_curr)
            acc_stu += (outputs_stu.max(1)[1] == y_curr).float().sum()
            acc_ema += (outputs_ema.max(1)[1] == y_curr).float().sum()

    return acc_stu.item() / x.shape[0],acc_ema.item() / x.shape[0]
    # acc_stu = 0.
    # # acc_ema = 0.
    # n_batches = math.ceil(x.shape[0] / batch_size)
    # with torch.no_grad():
    #     for counter in range(n_batches):
    #         x_curr = x[counter * batch_size:(counter + 1) *
    #                    batch_size].to(device)
    #         y_curr = y[counter * batch_size:(counter + 1) *
    #                    batch_size].to(device)

    #         outputs_stu = model(x_curr)
    #         acc_stu += (outputs_stu.max(1)[1] == y_curr).float().sum()
    #         # acc_ema += (outputs_ema.max(1)[1] == y_curr).float().sum()

    return acc_stu.item() / x.shape[0]
from pprint import pprint
if __name__ == '__main__':
    logger.info(vars(args))
    evaluate('"CIFAR-10-C evaluation.')
