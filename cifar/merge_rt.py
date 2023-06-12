import pandas as pd
import os

# base_dir=['/home/cll/Workspace/code/tta/cotta/cifar/results/cifar10-Standard/fixed_loc-12/N_4/sgd-lr_0.1-decay_1/bsz_100/trial_1/TTA',\
#     '/home/cll/Workspace/code/tta/cotta/cifar/results/cifar10-Standard/random_loc-7/N_1/sgd-lr_0.1-decay_1/bsz_100/trial_1/TTA',\
#      '/home/cll/Workspace/code/tta/cotta/cifar/results/cifar10-Standard/random_loc-7/N_4/sgd-lr_0.1-decay_1/bsz_100/trial_1/TTA'   ]
# folder_path = base_dir+'/pretrain'  # 替换为你的文件夹路径
# folder_path = base_dir # 替换为你的文件夹路径
base_dir='/home/cll/Workspace/code/tta/cotta/cifar/results/cifar10-Standard'
base_dir = '/home/cll/Workspace/code/tta/cotta/cifar/results3/cifar10-Standard'
# 获取base_dir目录下以及多级子目录下所有以 'TTA' 结尾的目录名
dirs = [x for x in os.walk(base_dir) if x[0].endswith('TTA')]
dfs = []  # 用于存储所有读取的DataFrame
# print(f'num of files:{len(os.listdir(folder_path))}')
num_files=0
for folder_path in dirs:
    folder_path=folder_path[0]
    num_files+=len(os.listdir(folder_path))
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") :
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except:
                pass
print(f'num of files:{num_files}')

merged_df = pd.concat(dfs, ignore_index=True)  
merged_df.dropna(axis=0, how='all', inplace=True)  # 删除空行
# save_path = base_dir+'/results'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# merged_df.to_csv(save_path+'/merged_pretrain.csv', index=False)# 合并所有DataFrame为一个DataFrame
merged_df.to_csv('results3'+'/merged_tta.csv', index=False)# 合并所有DataFrame为一个DataFrame
