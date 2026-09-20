"""Small-data diagnostic: python -m experiments.overfit --steps 2000."""
import argparse
import json
from pathlib import Path
import time

import torch
from datasets import load_from_disk
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw
from torchvision import transforms

from models.mini_flux import MiniFlux
from models.text_encoder import TextEncoder
from models.flow_matching import flow_matching_loss

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'data/coco')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/overfit')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--sample-every', type=int, default=250)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--time-scale', type=float, default=1000.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    args = parser.parse_args()
    if min(args.steps, args.sample_every, args.count, args.batch_size) < 1:
        parser.error('steps, sample-every, count and batch-size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.json').exists():
        parser.error('output already contains a run; choose a fresh --output directory')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    print(f'Device: {device}', flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['device'] = str(device)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    dataset = load_from_disk(str(args.dataset))
    if 'train' in dataset:
        dataset = dataset['train']
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.count, len(dataset))))
    transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    pixels = torch.stack([transform(row['image'].convert('RGB')) for row in dataset]).to(device)
    captions = list(dataset['text'])
    (args.output / 'captions.json').write_text(json.dumps(captions, indent=2))
    print('Loading VAE and CLIP; caching fixed latents and captions...', flush=True)
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device).eval().requires_grad_(False)
    encoder = TextEncoder(device)
    with torch.no_grad():
        latents = torch.cat([vae.encode(chunk).latent_dist.mode() for chunk in pixels.split(args.batch_size)]) * vae.config.scaling_factor
        text = encoder.encode(captions)
        recon = torch.cat([vae.decode(chunk / vae.config.scaling_factor).sample for chunk in latents.split(args.batch_size)])
    del encoder
    model = MiniFlux(time_scale=args.time_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Separate fixed sampling noise leaves the training RNG untouched during previews.
    fixed_noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(args.seed + 1)).to(device)

    def panel(generated, step):
        canvas = Image.new('RGB', (128*3, len(captions)*158), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, caption in enumerate(captions):
            for j, tensor in enumerate((pixels[i], recon[i], generated[i])):
                array = ((tensor.detach().cpu().clamp(-1, 1) + 1)*127.5).round().byte().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(array), (j*128, i*158))
            label = f'{i+1}: {caption}'.encode('ascii', 'replace').decode()[:60]
            draw.text((3, i*158+130), label, fill='black')
        canvas.save(args.output / f'samples_{step:06d}.png')

    @torch.no_grad()
    def sample(step):
        model.eval()
        outputs = []
        for noise, conditioning in zip(fixed_noise.split(args.batch_size), text.split(args.batch_size)):
            x = noise.clone()
            for s in range(50):
                t = torch.full((len(x),), s/50, device=device)
                x = x + model(x, conditioning, t)/50
            outputs.append(vae.decode(x / vae.config.scaling_factor).sample)
        panel(torch.cat(outputs), step)
        model.train()
        print(f'Saved sample grid at step {step}: original | VAE reconstruction | generated', flush=True)

    sample(0)
    start = time.monotonic()
    order = torch.empty(0, dtype=torch.long)
    running = 0.0
    with (args.output / 'loss.csv').open('w', buffering=1) as log:
        log.write('step,loss,elapsed_seconds\n')
        for step in range(1, args.steps+1):
            if not len(order):
                order = torch.randperm(len(latents))
            ids, order = order[:args.batch_size].to(device), order[args.batch_size:]
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, latents[ids], text[ids])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            value = loss.item()
            running += value
            log.write(f'{step},{value:.7f},{time.monotonic()-start:.2f}\n')
            if step % 50 == 0:
                print(f'Step {step}/{args.steps}: mean loss {running/50:.5f}', flush=True)
                running = 0.0
            if step % args.sample_every == 0 or step == args.steps:
                sample(step)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'config': config}, args.output / 'checkpoint.pt')
    print(f'Finished: {args.output}', flush=True)


if __name__ == '__main__':
    main()
