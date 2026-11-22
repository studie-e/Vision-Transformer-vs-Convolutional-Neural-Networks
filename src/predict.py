"""Single-image inference with the stored training config and checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from src.data import CLASSES, make_eval_transform
from src.evaluate import checked_config
from src.train import factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checked_config(state, state['config'])
    model = factory(config['model']['factory'])(config)
    model.load_state_dict(state['model'])
    model.to(args.device).eval()
    with Image.open(args.image) as original:
        rgb = original.convert('RGB')
        if rgb.size != (32, 32):
            raise ValueError('Expected a 32x32 image, matching the CIFAR-10 input protocol')
        tensor = make_eval_transform()(rgb).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0].cpu().tolist()
    label = max(range(len(probabilities)), key=probabilities.__getitem__)
    result = dict(label=label, class_name=CLASSES[label],
                  probabilities=dict(zip(CLASSES, probabilities)),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  device=args.device)
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
