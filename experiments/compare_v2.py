"""Matched v1/v2 experiment: 128 train, 32 held out, cached shared features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from datasets import load_from_disk
from diffusers import AutoencoderKL
from torchvision import transforms

from experiments.generalization import save_grid
from models.mini_flux import MiniFlux
from models.mini_flux_v2 import MiniFluxV2
from models.text_encoder import TextEncoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT.parent/'data/coco')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/compare_v2')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--sample-every', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    args = parser.parse_args()
    if min(args.steps, args.sample_every, args.batch_size) < 1:
        parser.error('step counts and batch size must be positive')
    if (args.output/'config.json').exists():
        parser.error('choose a fresh output directory')
    device = torch.device(args.device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    torch.set_num_threads(4)
    def sync():
        if device.type == 'mps': torch.mps.synchronize()
        if device.type == 'cuda': torch.cuda.synchronize()
    dataset = load_from_disk(str(args.dataset))
    if 'train' in dataset: dataset = dataset['train']
    if len(dataset) < 1000: parser.error('requires the original 1000-example dataset')
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    # Same training/held-out boundary as the preceding 900/100 experiment.
    indices = order[:128] + order[900:932]
    transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
    pixels = torch.stack([transform(dataset[i]['image'].convert('RGB')) for i in indices])
    captions = [dataset[i]['text'] for i in indices]
    hashes = [hashlib.sha256(p.numpy().tobytes()).hexdigest() for p in pixels]
    assert not set(hashes[:128]) & set(hashes[128:]), 'exact image overlap across split'
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(device=str(device), platform=platform.platform(), torch_version=str(torch.__version__), train_count=128, validation_count=32, resolution=128, time_scale=1000., lr=1e-4, sampling_steps=50)
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    (args.output/'split.json').write_text(json.dumps({'train_indices':indices[:128], 'validation_indices':indices[128:], 'captions':captions, 'dataset_fingerprint':dataset._fingerprint, 'exact_image_overlap':0}, indent=2))
    print(f'Device {device}; 128 train / 32 held out; caching shared features', flush=True)
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device).eval().requires_grad_(False)
    encoder = TextEncoder(device)
    text_parameter_count = sum(p.numel() for p in encoder.encoder.parameters())
    latents, text = [], []
    with torch.no_grad():
        for start in range(0,160,args.batch_size):
            latents.append((vae.encode(pixels[start:start+args.batch_size].to(device)).latent_dist.mode()*vae.config.scaling_factor).cpu())
            text.append(encoder.encode(captions[start:start+args.batch_size]).cpu())
        empty = encoder.encode(['']).cpu()
    latents, text = torch.cat(latents), torch.cat(text)
    del encoder
    torch.save({'latents':latents, 'text':text, 'empty_text':empty}, args.output/'features.pt')
    if device.type=='mps': torch.mps.empty_cache()
    preview = {'train':torch.arange(8), 'validation':torch.arange(128,136)}
    recons = {}
    with torch.no_grad():
        for split, ids in preview.items():
            recons[split] = torch.cat([vae.decode(latents[ix].to(device)/vae.config.scaling_factor).sample.cpu() for ix in ids.split(args.batch_size)])
    eval_rng = torch.Generator().manual_seed(args.seed+20)
    eval_noise = torch.randn(latents.shape, generator=eval_rng)
    eval_t = torch.rand(160, generator=eval_rng)
    fixed_noise = torch.randn((8,4,16,16),generator=torch.Generator().manual_seed(args.seed+30))
    summary = {}
    for name, constructor in [('v1', MiniFlux), ('v2', MiniFluxV2)]:
        out = args.output/name
        out.mkdir()
        torch.manual_seed(args.seed)
        model = constructor(time_scale=1000.).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        train_rng = torch.Generator().manual_seed(args.seed+40)
        shuffle_rng = torch.Generator().manual_seed(args.seed+10)
        print(f'{name}: {sum(p.numel() for p in model.parameters()):,} generator parameters', flush=True)

        @torch.no_grad()
        def generate(conditioning, noise):
            outputs, final_latents = [], []
            for cond, initial in zip(conditioning.split(args.batch_size),noise.split(args.batch_size)):
                x=initial.to(device).clone(); cond=cond.to(device)
                for step in range(50):
                    t=torch.full((len(x),),step/50,device=device)
                    x=x+model(x,cond,t)/50
                final_latents.append(x.cpu())
                outputs.append(vae.decode(x/vae.config.scaling_factor).sample.cpu())
            return torch.cat(outputs), torch.cat(final_latents)

        @torch.no_grad()
        def evaluate(step, caption_checks=False):
            model.eval()
            metrics={}
            modes=['correct','shuffled','empty'] if caption_checks else ['correct']
            for mode in modes:
                values=[]
                for start in range(0,160,args.batch_size):
                    ids=torch.arange(start,min(start+args.batch_size,160))
                    if mode=='correct': cond=text[ids]
                    elif mode=='empty': cond=empty.expand(len(ids),-1,-1)
                    else:
                        shuffled=torch.where(ids<128,(ids+1)%128,128+(ids-128+1)%32)
                        cond=text[shuffled]
                    noise=eval_noise[ids].to(device); target=latents[ids].to(device); t=eval_t[ids].to(device)
                    xt=(1-t[:,None,None,None])*noise+t[:,None,None,None]*target
                    error=(model(xt,cond.to(device),t)-(target-noise)).square().flatten(1).mean(1)
                    values.extend(error.cpu().tolist())
                metrics[mode]={'train':sum(values[:128])/128,'validation':sum(values[128:])/32}
            for split,ids in preview.items():
                reference_latents=None
                for mode in modes:
                    if mode=='correct': cond=text[ids]
                    elif mode=='empty': cond=empty.expand(len(ids),-1,-1)
                    else: cond=text[ids.roll(-1)]
                    images,z=generate(cond,fixed_noise)
                    save_grid(out/f'{split}_{mode}_{step:06d}.png',pixels[ids],recons[split],images,[captions[i] for i in ids.tolist()],f'{name} {split}: {mode} captions, step {step}')
                    if mode=='correct': reference_latents=z
                    else: metrics[mode][f'{split}_latent_mse_vs_correct']=(z-reference_latents).square().mean().item()
            model.train()
            return metrics

        started=time.monotonic()
        best=float('inf'); best_step=0
        with (out/'loss.csv').open('w',buffering=1) as log, (out/'evaluation.csv').open('w',buffering=1) as ev:
            writer=csv.writer(log); evaluator=csv.writer(ev)
            writer.writerow(['step','loss','elapsed_seconds'])
            evaluator.writerow(['step','train_fixed_loss','validation_fixed_loss'])
            metrics=evaluate(0)
            evaluator.writerow([0,metrics['correct']['train'],metrics['correct']['validation']])
            order=torch.empty(0,dtype=torch.long); running=0.; update_seconds=0.
            for step in range(1,args.steps+1):
                if not len(order): order=torch.randperm(128,generator=shuffle_rng)
                ids,order=order[:args.batch_size],order[args.batch_size:]
                sync(); tick=time.perf_counter()
                target=latents[ids].to(device); cond=text[ids].to(device)
                noise=torch.randn(target.shape,generator=train_rng).to(device)
                t=torch.rand(len(ids),generator=train_rng).to(device)
                xt=(1-t[:,None,None,None])*noise+t[:,None,None,None]*target
                optimizer.zero_grad(set_to_none=True)
                loss=(model(xt,cond,t)-(target-noise)).square().mean()
                if not torch.isfinite(loss): raise RuntimeError(f'{name}: nonfinite loss at {step}')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                sync(); update_seconds+=time.perf_counter()-tick
                value=loss.item(); running+=value
                writer.writerow([step,value,time.monotonic()-started])
                if step%250==0:
                    print(f'{name} {step}/{args.steps}: mean loss {running/250:.5f}',flush=True); running=0.
                if step%args.sample_every==0 or step==args.steps:
                    metrics=evaluate(step,caption_checks=step==args.steps)
                    tr,vl=metrics['correct']['train'],metrics['correct']['validation']
                    evaluator.writerow([step,tr,vl])
                    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'architecture':name,'config':config}
                    torch.save(state,out/'checkpoint.pt')
                    if vl<best:
                        best=vl; best_step=step; torch.save(state,out/'best_validation.pt')
                    print(f'{name} evaluation {step}: train {tr:.5f}, validation {vl:.5f}',flush=True)
            del state
        model.eval()
        # Warm up, then time 4 images: 50 Euler steps + VAE decode + CPU copy.
        generate(text[:4],fixed_noise[:4]); sync()
        timings=[]
        for _ in range(3):
            tick=time.perf_counter(); generate(text[:4],fixed_noise[:4]); sync(); timings.append(time.perf_counter()-tick)
        total=sum(p.numel() for p in model.parameters())
        active=sum(p.numel() for p in model.parameters() if p.grad is not None)
        summary[name]={'generator_parameters':total,'gradient_connected_parameters':active,'full_pipeline_parameters':total+text_parameter_count+sum(p.numel() for p in vae.parameters()),'generator_weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'training_update_seconds':update_seconds,'elapsed_seconds':time.monotonic()-started,'four_image_generation_seconds_median':statistics.median(timings),'generation_timing_scope':'cached text embeddings; 50 Euler steps + VAE decode + CPU output; 4 images; excludes text encoder','best_validation_step':best_step,'best_validation_fixed_loss':best,'final_caption_checks':metrics}
        (out/'summary.json').write_text(json.dumps(summary[name],indent=2))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        del model,optimizer
        if device.type=='mps': torch.mps.empty_cache()
    print('Comparison finished: '+str(args.output),flush=True)


if __name__=='__main__': main()
