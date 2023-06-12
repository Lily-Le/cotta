# Reference https://github.com/hjbahng/visual_prompting/blob/main/models/prompters.py
import torch
import torch.nn as nn
import numpy as np

class Prompter(nn.Module):
    def __init__(self, args):
        super(Prompter, self).__init__()
        self.prompt_size = [int(args.prompt_size.split('_')[0]),int(args.prompt_size.split('_')[1])]
        self.base_size = args.image_size
        self.prompt= nn.Parameter(torch.randn([1, 3, self.prompt_size[0], self.prompt_size[1]]))


    def forward(self, x):
        base = torch.zeros(x.size(0), 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat(x.size(0) * [self.prompt]).to(x.device)
        base[:,:,0:self.prompt_size[0],0:self.prompt_size[1]]=base[:,:,0:self.prompt_size[0],0:self.prompt_size[1]]+prompt
        return x + base
    
class RandomPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        # END: 8f9c3d6f9d1c
        A = np.random.randint(self.isize - self.psize, size=(x.size(0), 2))
        prompt = torch.zeros([x.size(0), 3, self.isize, self.isize]).cuda()
        for i in range(x.size(0)):
            x_, y_ = A[i]
            prompt[i, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch
        return x + prompt
    
class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt
    

def fixed_loc(args):
    return Prompter(args)
def random_loc(args):
    return RandomPrompter(args)

def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)