import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Choose an encoder for the model.')
parser.add_argument('--encoder', type=str, choices=['vgg16', 'mobilenetv3', 'efficientnetv2'], default='vgg16', help='Encoder type')
args = parser.parse_args()

# Configuration for model encoder
chosen_encoder = args.encoder