"""Train a sub-10M Stable Diffusion LoRA on the local captioned images."""
import argparse, csv, json, random, time
from pathlib import Path

import torch
from datasets import load_from_disk
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]
MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"


def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def sample(pipe,prompts,seeds,path):
    pipe.set_progress_bar_config(disable=True); images=[]
    for prompt,seed in zip(prompts,seeds):
        generator=torch.Generator(device=pipe.device).manual_seed(seed)
        images.append(pipe(prompt,height=512,width=512,num_inference_steps=30,
                           guidance_scale=7.5,generator=generator).images[0])
    w,h=images[0].size; canvas=images[0].copy().resize((2*w,2*h))
    for i,image in enumerate(images): canvas.paste(image,((i%2)*w,(i//2)*h))
    canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=ROOT.parent/"data/coco")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/sd15_lora_smoke")
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--sample-every",type=int,default=100)
    p.add_argument("--rank",type=int,default=4)
    p.add_argument("--resolution",type=int,default=256)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--gradient-accumulation",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if not torch.cuda.is_available(): p.error("CUDA is required")
    if a.output.exists() and any(a.output.iterdir()): p.error("choose a fresh output directory")
    a.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe=StableDiffusionPipeline.from_pretrained(MODEL,torch_dtype=dtype).to("cuda")
    pipe.safety_checker=None
    pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False); pipe.unet.requires_grad_(False)
    pipe.unet.add_adapter(LoraConfig(
        r=a.rank,lora_alpha=a.rank,init_lora_weights="gaussian",
        target_modules=["to_q","to_k","to_v","to_out.0"]))
    pipe.unet.enable_gradient_checkpointing()
    trainable=parameter_count(pipe.unet)
    if trainable>=10_000_000: raise RuntimeError(f"{trainable:,} trainable parameters")

    dataset=load_from_disk(str(a.dataset)); dataset=dataset["train"] if "train" in dataset else dataset
    indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(a.seed))[:1000].tolist()
    train_ids,val_ids=indices[:900],indices[900:]
    transform=transforms.Compose([
        transforms.Resize(a.resolution,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(a.resolution),transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    prompts=[dataset[i]["text"] for i in val_ids[:4]]; seeds=[a.seed+100+i for i in range(4)]
    sample(pipe,prompts,seeds,a.output/"samples_000000.png")

    scheduler=DDPMScheduler.from_config(pipe.scheduler.config)
    params=[x for x in pipe.unet.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.learning_rate,weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    rng=torch.Generator().manual_seed(a.seed+1); order=torch.empty(0,dtype=torch.long)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(model=MODEL,precision=str(dtype),trainable_parameters=trainable,
                  train_indices=train_ids,validation_indices=val_ids,
                  sample_prompts=prompts,sample_seeds=seeds)
    (a.output/"config.json").write_text(json.dumps(config,indent=2))
    started=time.monotonic(); peak=0
    with (a.output/"loss.csv").open("w",buffering=1) as file:
        log=csv.writer(file); log.writerow(["step","loss","elapsed_seconds"]); pipe.unet.train()
        for step in range(1,a.steps+1):
            optimizer.zero_grad(set_to_none=True); total=0.
            for _ in range(a.gradient_accumulation):
                if not len(order): order=torch.randperm(900,generator=rng)
                position,order=order[0].item(),order[1:]; row=dataset[train_ids[position]]
                pixel=transform(row["image"].convert("RGB")).unsqueeze(0).to("cuda",dtype=dtype)
                tokens=pipe.tokenizer(row["text"],padding="max_length",truncation=True,
                    max_length=pipe.tokenizer.model_max_length,return_tensors="pt").input_ids.to("cuda")
                with torch.no_grad():
                    latent=pipe.vae.encode(pixel).latent_dist.sample()*pipe.vae.config.scaling_factor
                    text=pipe.text_encoder(tokens)[0]
                    noise=torch.randn(latent.shape,device="cuda",dtype=dtype)
                    timestep=torch.randint(0,scheduler.config.num_train_timesteps,(1,),device="cuda")
                    noisy=scheduler.add_noise(latent,noise,timestep)
                with torch.autocast("cuda",dtype=dtype):
                    prediction=pipe.unet(noisy,timestep,text).sample
                    loss=torch.nn.functional.mse_loss(prediction.float(),noise.float())
                scaler.scale(loss/a.gradient_accumulation).backward()
                total+=loss.item()/a.gradient_accumulation
            scaler.step(optimizer); scaler.update()
            peak=max(peak,torch.cuda.max_memory_allocated()); log.writerow([step,total,time.monotonic()-started])
            if step%20==0: print(f"step {step}/{a.steps}: loss {total:.5f}",flush=True)
            if step%a.sample_every==0 or step==a.steps:
                pipe.unet.eval(); sample(pipe,prompts,seeds,a.output/f"samples_{step:06d}.png"); pipe.unet.train()
                StableDiffusionPipeline.save_lora_weights(
                    a.output/f"lora_{step:06d}",
                    unet_lora_layers=get_peft_model_state_dict(pipe.unet))
                torch.save({"step":step,"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                            "rng":rng.get_state(),"order":order},a.output/"training_state.pt")
    (a.output/"summary.json").write_text(json.dumps({
        "steps":a.steps,"trainable_parameters":trainable,"peak_cuda_bytes":peak,
        "elapsed_seconds":time.monotonic()-started},indent=2))


if __name__=="__main__": main()
