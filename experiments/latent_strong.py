"""Train strongly text-conditioned MiniFlux in frozen 128px VAE latent space."""
import argparse,csv,hashlib,json,time
from pathlib import Path

import torch
from datasets import load_from_disk
from diffusers import AutoencoderKL
from torchvision import transforms

from experiments.generalization import save_grid
from models.cifar_reference import reference_components
from models.strong_latent_miniflux import StrongLatentMiniFlux
from models.text_encoder import TextEncoder
from utils.device import empty_device_cache,get_device

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT.parent/'data/coco')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/latent_strong_smoke')
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--sample-every',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--conditioning-dropout',type=float,default=.15)
    p.add_argument('--guidance-scale',type=float,default=3.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default=None)
    a=p.parse_args()
    if a.steps<1 or a.sample_every<1 or a.microbatch<1 or a.batch_size%a.microbatch: p.error('positive steps and divisible microbatch required')
    if not 0<=a.conditioning_dropout<=1: p.error('conditioning dropout must be in [0,1]')
    if (a.output/'config.json').exists(): p.error('choose a fresh output directory')
    device=get_device(a.device); torch.manual_seed(a.seed); torch.set_num_threads(4)
    dataset=load_from_disk(str(a.dataset)); dataset=dataset['train'] if 'train' in dataset else dataset
    if len(dataset)<1000: p.error('requires 1000 captioned images')
    ids=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(a.seed))[:1000].tolist()
    transform=transforms.Compose([transforms.Resize(128),transforms.CenterCrop(128),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    pixels=torch.stack([transform(dataset[i]['image'].convert('RGB')) for i in ids])
    captions=[dataset[i]['text'] for i in ids]
    hashes=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in pixels]
    if set(hashes[:900])&set(hashes[900:]): raise ValueError('exact image overlap across split')
    a.output.mkdir(parents=True); config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(device=str(device),resolution=128,latent_shape=[4,16,16],train_count=900,validation_count=100,precision='bfloat16' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'float32',vae='stabilityai/sd-vae-ft-mse',text_encoder='openai/clip-vit-base-patch32')
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    (a.output/'split.json').write_text(json.dumps({'indices':ids,'dataset_fingerprint':dataset._fingerprint,'exact_overlap':0},indent=2))
    print(f'Device {device}; caching VAE latents and CLIP features',flush=True)
    vae=AutoencoderKL.from_pretrained(config['vae']).to(device).eval().requires_grad_(False)
    encoder=TextEncoder(device); latents=[]; text=[]
    with torch.no_grad():
        for start in range(0,1000,a.batch_size):
            chunk=pixels[start:start+a.batch_size].to(device)
            latents.append((vae.encode(chunk).latent_dist.mode()*vae.config.scaling_factor).cpu())
            text.append(encoder.encode(captions[start:start+a.batch_size]).cpu())
            if start%160==0: print(f'cached {start}/1000',flush=True)
        empty=encoder.encode(['']).cpu()
    latents=torch.cat(latents); text=torch.cat(text); del encoder; empty_device_cache(device)
    torch.save({'latents':latents,'text':text,'empty_text':empty},a.output/'features.pt')
    preview_ids=torch.arange(8); fixed_noise=torch.randn((8,4,16,16),generator=torch.Generator().manual_seed(a.seed+30))
    with torch.no_grad():
        recon=torch.cat([vae.decode(latents[ix].to(device)/vae.config.scaling_factor).sample.cpu() for ix in preview_ids.split(a.microbatch)])
    save_grid(a.output/'vae_reconstruction.png',pixels[preview_ids],recon,recon,[captions[i] for i in preview_ids.tolist()],'VAE quality gate (generated column repeats reconstruction)')
    torch.manual_seed(a.seed); base=StrongLatentMiniFlux().to(device); EMA=reference_components()[2]; model=EMA(base).to(device)
    optimizer=torch.optim.AdamW(base.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01)
    use_amp=device.type=='cuda'; amp_dtype=torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp and amp_dtype==torch.float16)
    train_rng=torch.Generator().manual_seed(a.seed+10); shuffle_rng=torch.Generator().manual_seed(a.seed+11)
    eval_rng=torch.Generator().manual_seed(a.seed+20); eval_ids=torch.arange(900,1000)
    eval_noise=torch.randn((100,4,16,16),generator=eval_rng); eval_t=torch.rand(100,generator=eval_rng)

    @torch.no_grad()
    def generate(condition):
        model.eval(); outputs=[]
        for cond,initial in zip(condition.split(a.microbatch),fixed_noise.split(a.microbatch)):
            x=initial.to(device); cond=cond.to(device); null=empty.expand(len(x),-1,-1).to(device)
            for step in range(30):
                left=step/30; right=(step+1)/30; dt=right-left
                tl=torch.full((len(x),),left,device=device); tr=torch.full((len(x),),right,device=device)
                vc=model(x,cond,tl); vu=model(x,null,tl); velocity=vu+a.guidance_scale*(vc-vu)
                predictor=x+dt*velocity
                vc=model(predictor,cond,tr); vu=model(predictor,null,tr); correction=vu+a.guidance_scale*(vc-vu)
                x=x+dt*(velocity+correction)/2
            outputs.append(vae.decode(x/vae.config.scaling_factor).sample.cpu())
        model.train(True); return torch.cat(outputs)

    @torch.no_grad()
    def evaluate(step):
        model.eval(); losses=[]; deltas=[]
        for start in range(0,100,a.microbatch):
            ix=eval_ids[start:start+a.microbatch]; target=latents[ix].to(device); noise=eval_noise[start:start+len(ix)].to(device); ts=eval_t[start:start+len(ix)].to(device)
            xt=(1-ts[:,None,None,None])*noise+ts[:,None,None,None]*target; cond=text[ix].to(device); null=empty.expand(len(ix),-1,-1).to(device)
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
                prediction=model(xt,cond,ts); null_prediction=model(xt,null,ts)
            losses.extend((prediction.float()-(target-noise)).square().flatten(1).mean(1).cpu().tolist())
            deltas.extend((prediction.float()-null_prediction.float()).abs().flatten(1).mean(1).cpu().tolist())
        generated=generate(text[preview_ids]); save_grid(a.output/f'samples_{step:06d}.png',pixels[preview_ids],recon,generated,[captions[i] for i in preview_ids.tolist()],f'strong latent CFG {a.guidance_scale:g}, step {step}')
        return sum(losses)/len(losses),sum(deltas)/len(deltas)

    order=torch.empty(0,dtype=torch.long); started=time.monotonic(); peak=0
    with (a.output/'loss.csv').open('w',buffering=1) as lf,(a.output/'evaluation.csv').open('w',buffering=1) as ef:
        log,ev=csv.writer(lf),csv.writer(ef); log.writerow(['step','loss','elapsed_seconds']); ev.writerow(['step','validation_velocity_mse','conditional_null_velocity_mae'])
        validation,delta=evaluate(0); ev.writerow([0,validation,delta]); print(f'evaluation 0: {validation:.5f}',flush=True)
        for step in range(1,a.steps+1):
            if len(order)<a.batch_size: order=torch.cat((order,torch.randperm(900,generator=shuffle_rng)))
            batch,order=order[:a.batch_size],order[a.batch_size:]; optimizer.zero_grad(set_to_none=True); value=0.
            noise=torch.randn(latents[batch].shape,generator=train_rng); ts=torch.rand(len(batch),generator=train_rng); drop=torch.rand(len(batch),generator=train_rng)<a.conditioning_dropout
            conditioning=text[batch].clone(); conditioning[drop]=empty
            for start in range(0,len(batch),a.microbatch):
                target=latents[batch[start:start+a.microbatch]].to(device); n=noise[start:start+a.microbatch].to(device); t=ts[start:start+a.microbatch].to(device); cond=conditioning[start:start+a.microbatch].to(device)
                xt=(1-t[:,None,None,None])*n+t[:,None,None,None]*target
                with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp): loss=(model(xt,cond,t).float()-(target-n)).square().mean()/ (a.batch_size/a.microbatch)
                if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
                scaler.scale(loss).backward(); value+=loss.item()
            scaler.step(optimizer); scaler.update(); model.update_ema(); log.writerow([step,value,time.monotonic()-started])
            if device.type=='cuda': peak=max(peak,torch.cuda.max_memory_allocated())
            if step%20==0: print(f'step {step}/{a.steps}: loss {value:.5f}',flush=True)
            if step%a.sample_every==0 or step==a.steps:
                validation,delta=evaluate(step); ev.writerow([step,validation,delta])
                state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step,'train_rng':train_rng.get_state(),'shuffle_rng':shuffle_rng.get_state(),'config':config,'validation_velocity_mse':validation,'conditional_null_velocity_mae':delta}
                torch.save(state,a.output/'checkpoint.pt'); print(f'evaluation {step}: mse {validation:.5f}, cond/null {delta:.5f}',flush=True)
    (a.output/'summary.json').write_text(json.dumps({'steps':a.steps,'parameters':sum(p.numel() for p in base.parameters()),'validation_velocity_mse':validation,'conditional_null_velocity_mae':delta,'peak_cuda_bytes':peak,'elapsed_seconds':time.monotonic()-started},indent=2))


if __name__=='__main__': main()
