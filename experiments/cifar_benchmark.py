"""Time-bounded, resumable CIFAR-10 reference/MiniFlux flow benchmark."""
import argparse
import csv
import gc
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from models.cifar_reference import REFERENCE, COMMIT, reference_components
from models.cifar_miniflux import CifarMiniFlux
from models.cifar_hierarchical_miniflux import HierarchicalCifarMiniFlux

ROOT=Path(__file__).resolve().parents[1]


def skewed_timesteps(n, generator):
    return (1/(1+torch.exp(torch.randn(n,generator=generator)*1.2-1.2))).clamp(.0001,1.)


@torch.no_grad()
def heun_sample(model, initial, grid):
    x=initial.clone()
    for left,right in zip(grid[:-1],grid[1:]):
        left,right=float(left),float(right)
        dt=right-left
        velocity=model(x,torch.full((len(x),),left,device=x.device),extra={})
        predictor=x+dt*velocity
        velocity2=model(predictor,torch.full((len(x),),right,device=x.device),extra={})
        x=x+dt*(velocity+velocity2)/2
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/cifar10')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/cifar_benchmark')
    p.add_argument('--budget-seconds',type=float,default=1800)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    p.add_argument('--models',nargs='+',choices=['reference','miniflux','hierarchical'],default=['reference','miniflux'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-updates',type=int,default=0,help='Optional smoke-test cap per model')
    p.add_argument(
        '--target-epochs',
        type=float,
        default=0.0,
        help='Optional cumulative epoch target per model; zero disables it',
    )
    a=p.parse_args()
    if a.microbatch<1 or a.batch_size%a.microbatch or a.budget_seconds<30:
        p.error('budget >=30 seconds and batch size divisible by positive microbatch required')
    if len(set(a.models))!=len(a.models): p.error('models must be unique')
    if (a.output/'config.json').exists() and not a.resume: p.error('choose a fresh output or --resume')
    actual_commit=subprocess.check_output(['git','-C',str(REFERENCE),'rev-parse','HEAD'],text=True).strip()
    if actual_commit!=COMMIT: raise RuntimeError('Reference checkout does not match pinned revision')
    torch.set_num_threads(4)
    device=torch.device(a.device)
    train=CIFAR10(str(a.data),train=True,download=False)
    test=CIFAR10(str(a.data),train=False,download=False)
    train_pixels=torch.from_numpy(train.data).permute(0,3,1,2).contiguous()
    test_pixels=torch.from_numpy(test.data).permute(0,3,1,2).contiguous()
    eval_ids=torch.randperm(len(test),generator=torch.Generator().manual_seed(a.seed+1))[:128]
    eval_data=test_pixels[eval_ids].float()/127.5-1
    eval_rng=torch.Generator().manual_seed(a.seed+2)
    eval_noise=torch.randn(eval_data.shape,generator=eval_rng)
    eval_t=skewed_timesteps(len(eval_data),eval_rng)
    fixed_noise=torch.randn((16,3,32,32),generator=torch.Generator().manual_seed(a.seed+3))
    cls,reference_config,EMA,schedule,PathClass=reference_components()
    path=PathClass()
    # Upstream schedule has 50 intervals: Heun evaluates the model twice per interval.
    # Float32 avoids MPS float64 restrictions without changing the schedule's math.
    grid=schedule(nfes=50).float()
    a.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(reference_commit=COMMIT,reference_config=reference_config,platform=platform.platform(),torch_version=str(torch.__version__),train_examples=len(train),test_examples=len(test),evaluation_test_indices=eval_ids.tolist(),optimizer={'lr':1e-4,'betas':[.9,.95],'weight_decay':.01},sampling_intervals=50,sampling_model_evaluations=100,sampling_start_time=float(grid[0]),precision='float32',conditioning='unconditional')
    if a.resume:
        original=json.loads((a.output/'config.json').read_text())
        for key in ('seed','batch_size','microbatch','reference_commit'):
            if original[key]!=config[key]: raise ValueError(f'Resume mismatch: {key}')
    else: (a.output/'config.json').write_text(json.dumps(config,indent=2))
    summary_path=a.output/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    def sync():
        if device.type=='mps': torch.mps.synchronize()
        elif device.type=='cuda': torch.cuda.synchronize()
    benchmark_started=time.monotonic()
    global_deadline=benchmark_started+a.budget_seconds
    print(f'Benchmark: {a.budget_seconds:.0f}s total budget; full CIFAR-10; {device}',flush=True)
    for model_index,name in enumerate(a.models):
        # Includes model initialization, training, periodic evaluation and checkpoint I/O.
        remaining=global_deadline-time.monotonic()
        phase_deadline=time.monotonic()+remaining/(len(a.models)-model_index)
        out=a.output/name; out.mkdir(exist_ok=True)
        torch.manual_seed(a.seed)
        constructors={
            'reference': lambda: cls(**reference_config),
            'miniflux': CifarMiniFlux,
            'hierarchical': HierarchicalCifarMiniFlux,
        }
        base=constructors[name]().to(device)
        model=EMA(base).to(device)
        optimizer=torch.optim.AdamW(base.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01)
        train_rng=torch.Generator().manual_seed(a.seed+10)
        shuffle_rng=torch.Generator().manual_seed(a.seed+11)
        order=torch.empty(0,dtype=torch.long)
        step=0; exposures=0; train_seconds=0.; sampled_memory=0
        checkpoint=out/'checkpoint.pt'
        if a.resume and checkpoint.exists():
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            for opt in optimizer.state.values():
                for key,value in opt.items():
                    if isinstance(value,torch.Tensor) and key!='step': opt[key]=value.to(device)
            step=state['step']; exposures=state['exposures']; order=state['order']
            train_rng.set_state(state['train_rng']); shuffle_rng.set_state(state['shuffle_rng'])
            torch.set_rng_state(state['torch_rng'])
            if device.type=='mps' and 'mps_rng' in state: torch.mps.set_rng_state(state['mps_rng'])
            if device.type=='cuda' and 'cuda_rng' in state: torch.cuda.set_rng_state(state['cuda_rng'])
            train_seconds=state['train_seconds']; del state
        start_step=step; phase_start=time.monotonic()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        def memory_sample():
            nonlocal sampled_memory
            if device.type=='mps': sampled_memory=max(sampled_memory,torch.mps.current_allocated_memory())
            elif device.type=='cuda': sampled_memory=max(sampled_memory,torch.cuda.max_memory_allocated())
        @torch.no_grad()
        def evaluate():
            model.eval()
            loss_sum=0.
            for start in range(0,len(eval_data),a.microbatch):
                target=eval_data[start:start+a.microbatch].to(device)
                noise=eval_noise[start:start+a.microbatch].to(device)
                t=eval_t[start:start+a.microbatch].to(device)
                sampled=path.sample(x_0=noise,x_1=target,t=t)
                value=(model(sampled.x_t,t,extra={})-sampled.dx_t).square().flatten(1).mean(1)
                loss_sum+=value.sum().item(); memory_sample()
            images=[]; tick=time.monotonic()
            for chunk in fixed_noise.split(a.microbatch):
                images.append(heun_sample(model,chunk.to(device),grid).cpu()); memory_sample()
            sync(); sample_seconds=time.monotonic()-tick
            images=torch.cat(images)
            if not torch.isfinite(images).all(): raise RuntimeError('Non-finite generated samples')
            save_image((images.clamp(-1,1)+1)/2,out/f'samples_{step:07d}.png',nrow=4)
            torch.save(images,out/f'samples_{step:07d}.pt')
            model.train(True)
            return loss_sum/len(eval_data),sample_seconds
        def save_checkpoint():
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'exposures':exposures,'train_seconds':train_seconds,'order':order,'train_rng':train_rng.get_state(),'shuffle_rng':shuffle_rng.get_state(),'torch_rng':torch.get_rng_state(),'architecture':name,'config':config}
            if device.type=='mps': state['mps_rng']=torch.mps.get_rng_state()
            if device.type=='cuda': state['cuda_rng']=torch.cuda.get_rng_state()
            temp=out/'checkpoint.tmp'; torch.save(state,temp); temp.replace(checkpoint)
        mode='a' if a.resume and (out/'loss.csv').exists() else 'w'
        parameter_count=sum(p.numel() for p in base.parameters())
        print(f'{name}: {parameter_count:,} parameters; phase budget {phase_deadline-time.monotonic():.0f}s; starting step {step}',flush=True)
        with (out/'loss.csv').open(mode,buffering=1) as lf, (out/'evaluation.csv').open(mode,buffering=1) as ef:
            log,ev=csv.writer(lf),csv.writer(ef)
            if mode=='w':
                log.writerow(['step','loss','exposures','training_seconds'])
                ev.writerow(['step','ema_test_velocity_mse','sample16_seconds','phase_elapsed_seconds'])
            eval_start=time.monotonic(); validation,sample_seconds=evaluate(); eval_duration=time.monotonic()-eval_start
            ev.writerow([step,validation,sample_seconds,time.monotonic()-phase_start])
            last_eval=step; max_update_seconds=0.; running=0.; logged=0
            while time.monotonic()<phase_deadline-max(30.,eval_duration*1.5+10.,max_update_seconds*2):
                if a.max_updates and step-start_step>=a.max_updates: break
                if a.target_epochs and exposures >= a.target_epochs * len(train_pixels): break
                tick=time.monotonic()
                if len(order)<a.batch_size:
                    # Keep all examples; complete the batch from a fresh permutation.
                    order=torch.cat([order,torch.randperm(len(train_pixels),generator=shuffle_rng)])
                ids,order=order[:a.batch_size],order[a.batch_size:]
                target=train_pixels[ids].float()/127.5-1
                flips=torch.rand(a.batch_size,generator=train_rng)<.5
                target[flips]=target[flips].flip(-1)
                noise=torch.randn(target.shape,generator=train_rng)
                t=skewed_timesteps(a.batch_size,train_rng)
                optimizer.zero_grad(set_to_none=True); value=0.
                for start in range(0,a.batch_size,a.microbatch):
                    data=target[start:start+a.microbatch].to(device); n=noise[start:start+a.microbatch].to(device); ts=t[start:start+a.microbatch].to(device)
                    sampled=path.sample(x_0=n,x_1=data,t=ts)
                    loss=(model(sampled.x_t,ts,extra={})-sampled.dx_t).square().mean()
                    if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite {name} loss at {step}')
                    (loss/(a.batch_size/a.microbatch)).backward()
                    value+=loss.item()/(a.batch_size/a.microbatch); memory_sample()
                optimizer.step(); model.update_ema(); sync(); memory_sample()
                elapsed=time.monotonic()-tick; train_seconds+=elapsed; max_update_seconds=max(max_update_seconds,elapsed)
                step+=1; exposures+=a.batch_size; running+=value; logged+=1
                log.writerow([step,value,exposures,train_seconds])
                if step%10==0:
                    print(f'{name} step {step}, epochs {exposures/50000:.3f}, loss {running/logged:.4f}, update {elapsed:.2f}s, remaining {max(0,phase_deadline-time.monotonic()):.0f}s',flush=True)
                    running=0.; logged=0
                if step%100==0 and phase_deadline-time.monotonic()>eval_duration*2+30:
                    tick=time.monotonic(); validation,sample_seconds=evaluate(); eval_duration=max(eval_duration,time.monotonic()-tick)
                    ev.writerow([step,validation,sample_seconds,time.monotonic()-phase_start]); last_eval=step
                    save_checkpoint()
                    print(f'{name} EMA evaluation {step}: {validation:.5f}',flush=True)
            if last_eval!=step:
                validation,sample_seconds=evaluate()
                ev.writerow([step,validation,sample_seconds,time.monotonic()-phase_start])
            save_checkpoint()
        result={'steps':step,'new_steps':step-start_step,'examples_seen':exposures,'equivalent_epochs':exposures/50000,'parameters':parameter_count,'float32_weight_bytes':parameter_count*4,'training_seconds_cumulative':train_seconds,'phase_seconds':time.monotonic()-phase_start,'final_ema_test_velocity_mse':validation,'sample16_seconds':sample_seconds,'sampling_model_evaluations':100,'sampled_allocated_tensor_bytes':sampled_memory,'memory_note':'MPS allocation sampled after microbatch backward/evaluation; includes model, EMA, optimizer and evaluation backups; lower bound on peak, excludes allocator cache. CUDA uses allocator peak.','status':'budget-limited pilot; not a published-quality reproduction'}
        summary[name]=result; summary_path.write_text(json.dumps(summary,indent=2))
        print(f'{name} finished: {json.dumps(result)}',flush=True)
        del base,model,optimizer
        gc.collect()
        if device.type=='mps': torch.mps.empty_cache()
    elapsed=time.monotonic()-benchmark_started
    print(f'Benchmark finished in {elapsed:.1f}s; checkpoints resumable',flush=True)


if __name__=='__main__': main()
