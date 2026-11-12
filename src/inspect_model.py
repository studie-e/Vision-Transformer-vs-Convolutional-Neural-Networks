"""Count a model and, optionally, search a ResNet width to match a ViT budget."""

import argparse
import json
from pathlib import Path

import yaml

from src.models.resnet import CifarResNet, count_parameters
from src.train import factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--target', type=int, help='Actual trainable parameter count from Lan Anh')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps({'factory': config['model']['factory'],
                      'parameters': count_parameters(factory(config['model']['factory'])(config))}))
    if args.target is not None:
        if args.target <= 0:
            parser.error('--target must be positive')
        candidates = []
        for width in range(4, 129):
            parameters = count_parameters(CifarResNet(width=width))
            gap = abs(parameters - args.target) / max(parameters, args.target)
            candidates.append({'width': width, 'parameters': parameters, 'relative_gap': gap})
        best = min(candidates, key=lambda row: row['relative_gap'])
        print(json.dumps({**best, 'within_10_percent': best['relative_gap'] <= 0.10}))


if __name__ == '__main__':
    main()
