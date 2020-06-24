from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np
import PIL
import skimage as sk
import skimage.measure
import torch
import torch.optim
import os

from third_party.models import *
from third_party.utils.denoising_utils import *

parser = argparse.ArgumentParser(description='PyTorch Denoising')

parser.add_argument("--ckpt", type=str, default="test", help="check point name")
parser.add_argument("--gpu", type=str, default="0", help="training device")

parser.add_argument("--image", type=str, default="F16", help="file name for test image, loaded from \data\denoising")
parser.add_argument('--nlevel', default=0.5, type=float, help='percentage of corrupted pixels')

parser.add_argument("--alg", type=str, default="sgd", help="optimization algorithm, sgd or adam")
parser.add_argument('--l1', action='store_true', help='loss function, default to be l2')
parser.add_argument('--lr', default=1, type=float, help='learning rate for network parameters (i.e. theta)')
parser.add_argument('--lr_c', default=500, type=float, help='learning rate for corruption parameters (i.e., g and h)')
parser.add_argument('--num_iter', default=150000, type=int, help='number of training iterations')
parser.add_argument('--width', default=128, type=int, help='network width')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # '0'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# Load image

img = PIL.Image.open('data/denoising/' + args.image + '.png')
img = crop_image(img, d=32)
img_np = pil_to_np(img)

# Add noise

img_noisy_np = sk.util.random_noise(img_np, mode='s&p', amount=args.nlevel)
img_cor_np = img_noisy_np - img_np

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)        

# Display groundtruth

plot_image_grid([img_np, img_noisy_np], nrow=2, factor=8);
plt.savefig('./checkpoint/'+args.ckpt+'_true.png', transparent = True, pad_inches=2)
plt.close()

# Network (exactly the same as the denoising DIP network, except with tunable width)

input_depth = 32 
n_channels = 3
    
skip_n33d = args.width 
skip_n33u = args.width 
skip_n11 = 4
num_scales = 5

net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales,
                                    num_channels_up =   [skip_n33u]*num_scales,
                                    num_channels_skip = [skip_n11]*num_scales, 
                                    upsample_mode='bilinear', downsample_mode='stride',
                                    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype) 
    
net_input = get_noise(input_depth, 'noise', (img.size[1], img.size[0])).type(dtype).detach()    

# Corruption parameterizaation

r_img_cor_p_torch = torch.zeros_like(img_noisy_torch).normal_()*1e-5
r_img_cor_n_torch = torch.zeros_like(img_noisy_torch).normal_()*1e-5
r_img_cor_p_torch.requires_grad=True
r_img_cor_n_torch.requires_grad=True              
                      
# Loss

if args.l1:
    criterion = torch.nn.L1Loss().type(dtype)
else:
    criterion = torch.nn.MSELoss().type(dtype)

# Optimizer

p = get_params('net', net, net_input)  # network parameters to be optimized
p_c = [r_img_cor_p_torch, r_img_cor_n_torch]  # corruption parameters to be optimized

if args.alg == 'adam':
    optimizer = torch.optim.Adam(p, lr=args.lr)
    optimizer_c = torch.optim.Adam(p_c, lr=args.lr_c)
elif args.alg == 'sgd':
    optimizer = torch.optim.SGD(p, lr=args.lr)
    optimizer_c = torch.optim.SGD(p_c, lr=args.lr_c)
else:
    assert False  

# Optimize

reg_noise_std = 1./30. 
show_every = 500
loss_history = []
psnr_history = []
psnr_best = 0
def closure(iterator):
    
    global psnr_best
    
    net_input_perturbed = net_input + torch.zeros_like(net_input).normal_(std=reg_noise_std)
    r_img_torch = net(net_input_perturbed)
    r_img_cor_torch = r_img_cor_p_torch **2 - r_img_cor_n_torch **2        
    r_img_noisy_torch = r_img_torch + r_img_cor_torch  
            
    total_loss = criterion(r_img_noisy_torch, img_noisy_torch)        
    total_loss.backward()
    
    if iterator % show_every == 0 or iterator == args.num_iter - 1:
        # evaluate recovered image
        r_img_np = torch_to_np(r_img_torch)
        # psnr = skimage.measure.compare_psnr(img_np, r_img_np) 
        psnr = skimage.metrics.peak_signal_noise_ratio(img_np, r_img_np) 

        print ('Iteration %05d    Loss %f   PSNR %f' % (iterator, total_loss.item(), psnr), '\n', end='')
        
        loss_history.append(total_loss.item())
        psnr_history.append(psnr)
                
        # save the best result to file
        if psnr > psnr_best:
            psnr_best = psnr
            plot_image_grid([np.clip(r_img_np, 0, 1)], factor=8, nrow=2)
            plt.savefig('./checkpoint/'+args.ckpt+'_best.png', transparent = True, pad_inches=2)
            plt.close()

            state = {'psnr': psnr,
                     'loss': total_loss.item(),
                     'r_img_np': r_img_np,
                     'iter': iterator}
            torch.save(state, './checkpoint/'+args.ckpt+'_best')        

        # save the last result and psnr/loss history to file
        if iterator == args.num_iter - 1:
            plot_image_grid([np.clip(r_img_np, 0, 1)], factor=8, nrow=2)
            plt.savefig('./checkpoint/'+args.ckpt+'_last.png', transparent = True, pad_inches=2)
            plt.close()

            state = {'psnr_history': psnr_history,
                     'loss_history': loss_history,
                     'r_img_np': r_img_np,
                     'iter': iterator}
            torch.save(state, './checkpoint/'+args.ckpt+'_last')        

    return total_loss


for iterator in range(args.num_iter):
    optimizer.zero_grad()
    optimizer_c.zero_grad()
    closure(iterator)
    optimizer.step()
    optimizer_c.step()         





