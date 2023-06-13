#!/bin/bash
# for lr in 1e-4 1e-5
# do 
# CUDA_VISIBLE_DEVICES=2 python cifar10c_prompt.py --cfg cfgs/cifar10/vpt.yaml --method fixed_loc --learning_rate $lr --optim sgd --N 1 --prompt_size 7 --resume output/cifar10-Standard/random_loc-7/N_1/sgd-lr_0.1-decay_1/bsz_100/trial_1/checkpoint.pth.tar &
# CUDA_VISIBLE_DEVICES=3 python cifar10c_prompt.py --cfg cfgs/cifar10/vpt.yaml --method fixed_loc --learning_rate $lr --optim sgd --N 4 --prompt_size 7 --resume output/cifar10-Standard/random_loc-7/N_4/sgd-lr_0.1-decay_1/bsz_100/trial_1/checkpoint.pth.tar &
# done

for a in 0.99
do
for lr in 1e-5 #1e-4 1e-3
do
CUDA_VISIBLE_DEVICES=3 python cifar10c_prompt.py --cfg cfgs/cifar10/vpt.yaml --method fixed_loc --mt_alpha $a --learning_rate $lr --optim adam --N 3 --prompt_size 2_5 --resume output3/cifar10-Standard/fixed_loc-2_5/N_3/mt_$a/adam-lr_0.1-decay_0/bsz_100/trial_1/3.pth.tar &
# CUDA_VISIBLE_DEVICES=3 python cifar10c_prompt.py --cfg cfgs/cifar10/vpt.yaml --method fixed_loc --mt_alpha $a --learning_rate $lr --optim sgd --N 5 --prompt_size 2_5 --resume output3/cifar10-Standard/fixed_loc-2_5/N_32/mt_$a/adam-lr_0.0001-decay_0/bsz_100/trial_1/9.pth.tar &
# CUDA_VISIBLE_DEVICES=0 python cifar10c_prompt.py --cfg cfgs/cifar10/vpt.yaml --method fixed_loc --mt_alpha $a --learning_rate $lr --optim sgd --N 4 --prompt_size 2_5 --resume output3/cifar10-Standard/fixed_loc-2_5/N_4/mt_$a/sgd-lr_0.1-decay_1/bsz_100/trial_1/model_best.pth.tar &

# done
# CUDA_VISIBLE_DEVICES=1 python main_clip.py --N 32 --prompt_size 2_5 --method fixed_loc --mt_alpha $a --learning_rate $lr &
# CUDA_VISIBLE_DEVICES=2 python main_clip.py --N 3 --prompt_size 2_5 --method fixed_loc --mt_alpha $a &
done
done



