from __future__ import print_function
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
import sys
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from vpt import VPT_wrapper,get_tta_transforms
from my_utils import Logger

from tensorboardX import SummaryWriter 
import json
    # create a summary writer object

# logger = logging.getLogger(__name__)

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
    parser.add_argument('--epochs', type=int, default=10, #default=3
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.1,   
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=0,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    # parser.add_argument('--model', type=str, default='clip')
    # parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='random_loc',
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
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')
    parser.add_argument('--N', type=int, default=1,
                        help='number of augmentations')
    parser.add_argument('--mt_alpha', type=float, default=0.99,
                        help='coefficient for ema')
    parser.add_argument('--cfg', dest='cfg_file',default='cfgs/cifar10/tent.yaml',
                        help='optional config file',)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    args = parser.parse_args()


    return args

best_acc1_ema = 0

best_acc1_stu = 0
best_acc1_val=0
device = "cuda" if torch.cuda.is_available() else "cpu"


args = parse_option()
    # args.model_folder = os.path.join('outputs', args.filename)
    # if not os.path.isdir(args.model_folder):
    #     os.makedirs(args.model_folder)

    
description='"CIFAR-10 Train Prompters.'
load_cfg_fom_args(args,description)
args.model_folder = args.save_path
sys.stdout = Logger(args.model_folder+'/train_prompt.log', sys.stdout)
args.log_path=args.save_path.replace('output','logs')
writer = SummaryWriter(args.log_path)
def main():
    global best_acc1_ema, best_acc1_stu, best_acc1_val,device



    # args.filename = '{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}'. \
    #     format(args.method, args.prompt_size, args.dataset, cfg.MODEL.ARCH,
    #            args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial)
    print (args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                        cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    
    for param in base_model.parameters():
        param.requires_grad = False
        
    prompter = prompters.__dict__[args.method](args).to(device)
    # 23456
    # create model
    # model, preprocess = clip.load('ViT-B/32', device, jit=False)
    # convert_models_to_fp32(model)
    # model.eval()



    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         prompter.load_state_dict(checkpoint['state_dict'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # create data

    # train_dataset = CIFAR100(args.root, transform=None,
    #                          download=True, train=True)

    # val_dataset = CIFAR100(args.root, transform=None,
    #                        download=True, train=False)

    # train_loader = DataLoader(train_dataset,
    #                           batch_size=args.batch_size, pin_memory=True,
    #                           num_workers=args.num_workers, shuffle=True)

    # val_loader = DataLoader(val_dataset,
    #                         batch_size=args.batch_size, pin_memory=True,
    #                         num_workers=args.num_workers, shuffle=False)

    # class_names = train_dataset.classes
    # class_names = refine_classname(class_names)
    # texts = [template.format(label) for label in class_names]
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    ## TTA  transform当成domain change,拿来作validation dataset()
    transform_test= get_tta_transforms()   
    transform_test.transforms.insert(0,transforms.ToTensor())
    # ddecorate里面有tta transform,只做tensor化
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    val_loader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # define criterion and optimizer


    model = VPT_wrapper(args,base_model,prompter)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    #GradScaler用于在训练过程中自动缩放梯度。
    #GradScaler会在每个训练迭代中计算梯度的范数，并根据范数的大小自动调整缩放因子。然后，`GradScaler` 将梯度乘以缩放因子，并将缩放后的梯度传递给优化器进行更新。

    # scaler = GradScaler()
    # total_steps = len(train_loader) * args.epochs
    # scheduler = cosine_lr(model.optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    # refined_template = template.lower().replace(' ', '_')
    # args.filename = f'{args.filename}_template_{refined_template}'

    

    # wandb
    # if args.use_wandb:
    #     wandb.init(project='Visual Prompting')
    #     wandb.config.update(args)
    #     wandb.run.name = args.filename
    #     wandb.watch(prompter, criterion, log='all', log_freq=10)




    epochs_since_improvement = 0


    # 记录每个epoch的train()和validate()的返回值
    # results = {'epoch': [], 'train_loss_ema': [], 'train_acc1_ema': [], 'train_loss_stu': [], 'train_acc1_stu': [], 'val_loss_stu': [], 'val_acc1_stu': []}
    args.start_epoch=0
    for epoch in range(args.start_epoch, args.epochs):
        # train()函数返回四个值，分别是losses_ema.avg, top1_ema.avg, losses_stu.avg, top1_stu.avg
        train_loss_ema, train_acc1_ema, train_loss_stu, train_acc1_stu = train(train_loader, model, criterion, epoch, args)
        # validate()函数返回两个值，分别是losses_stu.avg, top1_stu.avg
        val_loss_stu, val_acc1_stu = validate(val_loader, model, criterion, args, epoch)

        # 将train()和validate()的返回值打包成一个字典

        # 其他代码...

    # # 将结果写入json文件
    #     with open('results.json', 'w') as f:
    #         json.dump(results, f)

        # remember best Acc1_and save checkpoint
        is_best_ema = train_acc1_ema.avg > best_acc1_ema
        best_acc1_ema = max(train_acc1_ema.avg, best_acc1_ema)

        is_best_stu = train_acc1_stu.avg > best_acc1_stu
        best_acc1_stu = max(train_acc1_stu.avg, best_acc1_stu)

        is_best_val = val_acc1_stu.avg > best_acc1_val
        best_acc1_val = max(val_acc1_stu.avg , best_acc1_val)

        is_best = is_best_val
        if (epoch % 3 ==0) and (epoch !=0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'best_acc1_ema': best_acc1_ema,
                'best_acc1_stu': best_acc1_stu,
                'optimizer': model.optimizer.state_dict(),
            }, args, is_best=is_best,filename=f'{epoch}.pth.tar')

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break
    # losses_stu_test,acc1_stu_test = validate(val_loader,  model, criterion, args,epoch)
    # Convert any non-serializable objects to serializable objects
    # writer_dict = writer.__dict__
    # for key, value in writer_dict.items():
    #     if isinstance(value, torch.Tensor):
    #         writer_dict[key] = value.tolist()
    #     elif isinstance(value, CometLogger):
    #         writer_dict[key] = str(value)
    # # Write writer_dict to JSON file
    # with open(args.log_path+'/'+'writer.json', 'w') as f:
    #     json.dump(writer_dict, f)
    # wandb.run.finish()


def train(train_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ema = AverageMeter('Loss ema', ':.4e')
    top1_ema = AverageMeter('Acc1_ema', ':6.2f')
    losses_stu = AverageMeter('Loss stu', ':.4e')
    top1_stu = AverageMeter('Acc1_stu', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_ema, top1_ema,losses_stu,top1_stu],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.model[0].train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()

    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        # step = num_batches_per_epoch * epoch + i
        # scheduler(step)

        # optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)

        # with automatic mixed precision

        outputs_ema, outputs = model(images)
        loss_ema = criterion(outputs_ema, target)
        loss_stu = criterion(outputs,target)


        # Note: we clamp to 4.6052 = ln(100), as in the original paper.

        # measure accuracy
        acc1_ema = (outputs_ema.max(1)[1] == target).float().sum()
        acc1_stu = (outputs.max(1)[1] == target).float().sum()
        losses_ema.update(loss_ema.item(), images.size(0))
        top1_ema.update(acc1_ema.item(), images.size(0))
        losses_stu.update(loss_stu.item(),images.size(0))
        top1_stu.update(acc1_stu.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            # log to TensorBoard
            writer.add_scalar('Training/Loss_ema', losses_ema.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Training/Acc1_ema', top1_ema.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Training/Loss_stu', losses_stu.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Training/Acc1_stu', top1_stu.avg, epoch * len(train_loader) + i)

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'prompter_state_dict': model.model[0].state_dict(),
                'best_acc1_ema': best_acc1_ema,
                'best_acc1_stu': best_acc1_stu,
                'optimizer': model.optimizer.state_dict(),
            }, args)

    return losses_ema, top1_ema ,losses_stu,top1_stu


def validate(val_loader,model, criterion, args,epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses_stu = AverageMeter('Loss stu', ':.4e')
    top1_stu = AverageMeter('Acc1_stu', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time,losses_stu,top1_stu],
        prefix='Validate: ')


    # switch to evaluation mode
    model.model[0].eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)


            outputs = model(images)
            loss_stu = criterion(outputs,target)


            # Note: we clamp to 4.6052 = ln(100), as in the original paper.

            # measure accuracy

            acc1_stu = (outputs.max(1)[1] == target).float().sum()
            losses_stu.update(loss_stu.item(),images.size(0))
            top1_stu.update(acc1_stu.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Loss student {losses_stu.avg:.3f} Acc1_student {top1_stu.avg:.3f}'
                .format(losses_stu=losses_stu, top1_stu=top1_stu))

        # log to TensorBoard
        writer.add_scalar('Validation/Loss_stu', losses_stu.avg, epoch)
        writer.add_scalar('Validation/Acc1_stu', top1_stu.avg, epoch)

    return losses_stu,top1_stu


if __name__ == '__main__': 
    main()


    # close the summary writer object
    writer.close()
