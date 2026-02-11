"""
Evaluation script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from configs.model_config import ModelConfig
from models.vla_model import VLAModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    config = ModelConfig()
    model = VLAModel(config).to(args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    print("Model loaded successfully!")
    print("To evaluate in LIBERO simulator, install libero package")


if __name__ == '__main__':
    main()
