#!/usr/bin/env python3
import os
import argparse
import torch

from src.config import Config
from src.utils import set_seed
from src.cell_synthesis import train_model

import wandb 
import yaml


if __name__ == "__main__":
    
   run = wandb.init()
    
   #  run_id = run.id
   

   wandb.init(project="cell-synthesis_for_quality")

   # Configuration
   parser = argparse.ArgumentParser(description="Configuration")
   parser.add_argument('--config', type=str, default='default_config', help='configuration filename')
   parser.add_argument('--latent_dims', type=int)
   parser.add_argument('--map_layers', type=int)
   parser.add_argument('--gen_feat_num', type=int)
   parser.add_argument('--dis_feat_num', type=int)
   parser.add_argument('--init_type', type=str)
   parser.add_argument('--init_gain', type=float)
   parser.add_argument('--lr_g', type=float)
   parser.add_argument('--lr_d', type=float)
   parser.add_argument('--beta1', type=float)
   parser.add_argument('--beta2', type=float)
   parser.add_argument('--ema_decay', type=float)
   parser.add_argument('--ema_start', type=int)
   parser.add_argument('--gan_mode', type=str)
   parser.add_argument('--recon_mode', type=str)
   parser.add_argument('--d_r1_weight', type=float)
   parser.add_argument('--batch_size', type=int)
   parser.add_argument('--total_iters', type=int)
   args = parser.parse_args()
   
   
   config = Config(filename=args.config, mode='train')

   # Device
   os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
   os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
   torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
   sweep_config = wandb.config
   sweep_id = run.id
   
   # os.mkdir(f'/{sweep_id}')
   
   # config['LOG_PATH'] = f'/{sweep_id}/checkpoints/logs'
   # config['CKPT_PATH'] = f'/{sweep_id}/checkpoints/models'
   # config['SAMPLE_PATH'] = f'/{sweep_id}/checkpoints/samples'
   
   ## network
   if args.cond_size:
     config['COND_SIZE'] = args.cond_size
     
   if args.latent_dims:   
     config['LATENT_DIMS'] = args.latent_dims
     
   if args.map_layers:
     config['MAP_LAYERS'] = args.map_layers
     
   if args.gen_feat_num:
     config['GEN_FEAT_NUM'] = args.gen_feat_num
     
   if args.dis_feat_num:
     config['DIS_FEAT_NUM'] = args.dis_feat_num
     
   if args.init_type:
     config['INIT_TYPE'] = args.init_type
     
   if args.init_gain:
     config['INIT_GAIN'] = args.init_gain  
   
   # ## optimization
   if args.lr_g:
     config['LR_G'] = args.lr_g
     
   if args.lr_d:
     config['LR_D'] = args.lr_d
     
   if args.beta1:
     config['BETA1'] = args.beta1
     
   if args.beta2:
     config['BETA2'] = args.beta2
     
   if args.ema_decay:
     config['EMA_DECAY'] = args.ema_decay
     
   if args.ema_start:
     config['EMA_START'] = args.ema_start
     
   if args.gan_mode:
     config['GAN_MODE'] = args.gan_mode
     
   if args.recon_mode:
     config['RECON_MODE'] = args.recon_mode
     
   if args.d_r1_weight:
     config['D_R1_WEIGHT'] = args.d_r1_weight
   
   ## training  
   if args.batch_size:
     config['BATCH_SIZE'] = args.batch_size
     
   if args.total_iters:
     config['TOTAL_ITERS'] = args.total_iters

   # Random seed
   set_seed(config.MANUAL_SEED)
   
   config.update_config(config)
   

   # Main
   config.print_info()
   train_model(config)
    
   #  run.finish()
