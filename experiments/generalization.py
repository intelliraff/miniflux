"""Train a fresh MiniFlux with a reproducible 900/100 train/validation split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw
from torchvision import transforms

from models.flow_matching import flow_matching_loss
from models.mini_flux import MiniFlux
from models.text_encoder import TextEncoder

ROOT = Path(__file__).resolve().parents[1]


def save_grid(path, originals, reconstructions, generated, captions, title):
    canvas = Image.new('RGB', (600, 36 + len(captions) * 158), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), title, fill='black')
    draw.text((4, 18), 'Reference             VAE reconstruction    Generated', fill='black')
    for i, caption in enumerate(captions):
        top = 36 + i * 158
        for j, tensor in enumerate((originals[i], reconstructions[i], generated[i])):
            array = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array), (j * 200, top))
        label = caption.encode('ascii', 'replace').decode()[:95]
        draw.text((4, top + 132), label, fill='black')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'data/coco')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/generalization')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--sample-every', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time-scale', type=float, default=1000.)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    args = parser.parse_args()
    if min(args.steps, args.sample_every, args.batch_size) < 1:
        parser.error('step counts and batch size must be positive')
    if (args.output / 'config.json').exists():
        parser.error('choose a fresh output directory')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    dataset = load_from_disk(str(args.dataset))
    if 'train' in dataset:
        dataset = dataset['train']
    if len(dataset) < 1000:
        parser.error('this experiment requires at least 1000 examples')
    ids = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:1000].tolist()
    transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    pixels = torch.stack([transform(dataset[i]['image'].convert('RGB')) for i in ids])
    captions = [dataset[i]['text'] for i in ids]
    hashes = [hashlib.sha256(p.numpy().tobytes()).hexdigest() for p in pixels]
    overlap = set(hashes[:900]) & set(hashes[900:])
    if overlap:
        raise ValueError(f'{len(overlap)} identical image(s) cross the split; group duplicates before training')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), train_count=900, validation_count=100, resolution=128, lr=1e-4)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    (args.output / 'split.json').write_text(json.dumps({'dataset_fingerprint': dataset._fingerprint, 'train_indices': ids[:900], 'validation_indices': ids[900:], 'train_captions': captions[:900], 'validation_captions': captions[900:], 'exact_image_overlap': len(overlap)}, indent=2))
    print(f'Device: {device}; split: 900 train / 100 held out; no exact image overlap', flush=True)
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device).eval().requires_grad_(False)
    encoder = TextEncoder(device)
    latents, embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(pixels), args.batch_size):
            chunk = pixels[start:start+args.batch_size].to(device)
            latents.append((vae.encode(chunk).latent_dist.mode() * vae.config.scaling_factor).cpu())
            embeddings.append(encoder.encode(captions[start:start+args.batch_size]).cpu())
            if start % 100 == 0:
                print(f'Cached {start}/{len(pixels)} examples', flush=True)
    latents, embeddings = torch.cat(latents), torch.cat(embeddings)
    del encoder
    if device.type == 'mps':
        torch.mps.empty_cache()
    # Reset model initialization independently of dataset size and encoder loading.
    torch.manual_seed(args.seed)
    model = MiniFlux(time_scale=args.time_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Dedicated CPU generators keep evaluation from changing training randomness.
    shuffle_rng = torch.Generator().manual_seed(args.seed + 10)
    eval_rng = torch.Generator().manual_seed(args.seed + 20)
    eval_ids = torch.cat([torch.arange(100), torch.arange(900, 1000)])
    eval_noise = torch.randn((200, 4, 16, 16), generator=eval_rng)
    eval_t = torch.rand((200,), generator=eval_rng)
    preview_ids = {'train': torch.arange(8), 'validation': torch.arange(900, 908)}
    preview_noise = torch.randn((8, 4, 16, 16), generator=torch.Generator().manual_seed(args.seed + 30))
    best_val = float('inf')
    started = time.monotonic()

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        losses = []
        for start in range(0, len(eval_ids), args.batch_size):
            ix = eval_ids[start:start+args.batch_size]
            target = latents[ix].to(device)
            noise = eval_noise[start:start+len(ix)].to(device)
            t = eval_t[start:start+len(ix)].to(device)
            xt = (1-t[:, None, None, None])*noise + t[:, None, None, None]*target
            prediction = model(xt, embeddings[ix].to(device), t)
            losses.extend(F.mse_loss(prediction, target-noise, reduction='none').flatten(1).mean(1).cpu().tolist())
        for name, ix in preview_ids.items():
            generated, recon = [], []
            for start in range(0, len(ix), args.batch_size):
                batch_ids = ix[start:start+args.batch_size]
                text = embeddings[batch_ids].to(device)
                x = preview_noise[start:start+len(batch_ids)].to(device).clone()
                for s in range(50):
                    t = torch.full((len(x),), s/50, device=device)
                    x = x + model(x, text, t)/50
                generated.append(vae.decode(x/vae.config.scaling_factor).sample.cpu())
                recon.append(vae.decode(latents[batch_ids].to(device)/vae.config.scaling_factor).sample.cpu())
            save_grid(args.output / f'{name}_{step:06d}.png', pixels[ix], torch.cat(recon), torch.cat(generated), [captions[i] for i in ix.tolist()], f'{name} - step {step}')
        model.train()
        return sum(losses[:100])/100, sum(losses[100:])/100

    with (args.output / 'loss.csv').open('w', buffering=1) as train_log, (args.output / 'evaluation.csv').open('w', buffering=1) as eval_log:
        train_writer, eval_writer = csv.writer(train_log), csv.writer(eval_log)
        train_writer.writerow(['step', 'loss', 'elapsed_seconds'])
        eval_writer.writerow(['step', 'train_fixed_loss', 'validation_fixed_loss'])
        train_loss, val_loss = evaluate(0)
        eval_writer.writerow([0, train_loss, val_loss])
        order = torch.empty(0, dtype=torch.long)
        running = 0.
        for step in range(1, args.steps+1):
            if not len(order):
                order = torch.randperm(900, generator=shuffle_rng)
            ix, order = order[:args.batch_size], order[args.batch_size:]
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, latents[ix].to(device), embeddings[ix].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            value = loss.item()
            running += value
            train_writer.writerow([step, value, round(time.monotonic()-started, 2)])
            if step % 100 == 0:
                print(f'Step {step}/{args.steps}: mean training loss {running/100:.5f}', flush=True)
                running = 0.
            if step % args.sample_every == 0 or step == args.steps:
                train_loss, val_loss = evaluate(step)
                eval_writer.writerow([step, train_loss, val_loss])
                print(f'Evaluation {step}: train {train_loss:.5f}, held-out {val_loss:.5f}; grids saved', flush=True)
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'config': config, 'train_fixed_loss': train_loss, 'validation_fixed_loss': val_loss}
                temporary = args.output / 'checkpoint.tmp'
                torch.save(state, temporary)
                temporary.replace(args.output / 'checkpoint.pt')
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(state, args.output / 'best_validation.pt')
    print(f'Finished: {args.output}', flush=True)


if __name__ == '__main__':
    main()
