import os 
from glob import glob 
import argparse
from src.config import Config

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument('--config', type=str, default='default_config', help='configuration filename')
    args = parser.parse_args()
    config = Config(filename=args.config, mode='train')
    
    # Configuration
    # data_dir = 'data/celeba'
    # train_dir = os.path.join(data_dir, 'train')
    # test_dir = os.path.join(data_dir, 'test')
    
    # Train
    train_images = glob(os.path.join(config.DATAROOT, '*', '*.png'))
    with open(os.path.join(config.DATAROOT, 'list.txt'), 'w') as f:
        for ii in train_images:
            ii = ii.replace('\\', '/')
            f.write('/'.join(ii.split('/')[-2:]) + '\n')
    
    print('Done!')