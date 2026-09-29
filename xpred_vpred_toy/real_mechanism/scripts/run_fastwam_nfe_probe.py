#!/usr/bin/env python3
"""Real FastWAM action-only NFE probe on native three-view LeRobot manifests."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq, torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from real_mechanism.backends import NFE_GRID, SIGMA_GRID
from real_mechanism.data import load_manifest_actions

TOY = Path(__file__).resolve().parents[2]; FAST = TOY.parent / "FastWAM"

def _specs(values):
    out=[]
    for v in values:
        n,sep,p=v.partition("=")
        if not sep: raise ValueError("--checkpoint requires name=path")
        p=Path(p).expanduser().resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        out.append((n,p))
    return out

def _frame(video: Path, index: int):
    with av.open(str(video)) as c:
        return next(f for i,f in enumerate(c.decode(video=0)) if i==index).to_ndarray(format="rgb24")

def _observation(row):
    parquet=Path(row["parquet_path"]); root=parquet.parents[2]
    episode=int(Path(row["episode"]).stem.split("_")[-1]); anchor=int(row["anchor"])
    images={}
    for out,cam in (("head_camera","cam_high"),("left_camera","cam_left_wrist"),("right_camera","cam_right_wrist")):
        video=root/"videos"/f"chunk-{episode//1000:03d}"/f"observation.images.{cam}"/f"episode_{episode:06d}.mp4"
        images[out]={"rgb":_frame(video,anchor)}
    state=np.asarray(pq.read_table(parquet,columns=["observation.state"])["observation.state"].to_pylist()[anchor],np.float32)
    return {"observation":images,"joint_action":{"vector":state}},state

def _cfg(task):
    if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3",config_dir=str(FAST/"configs")):
        return compose(config_name="sim_robotwin.yaml",overrides=[f"task={task}"])

def _prediction_type(name: str, checkpoint: Path) -> str:
    """Read the type saved with a run; names are only a safe fallback."""
    from omegaconf import OmegaConf
    config = checkpoint.parents[2] / "config.yaml"
    if config.is_file():
        value = OmegaConf.select(OmegaConf.load(config), "model.action_prediction_type")
        if value is not None:
            return str(value)
    lower = name.lower()
    if "vloss" in lower or "v_loss" in lower:
        return "x_prediction_v_loss"
    if "xpred" in lower or "x_prediction" in lower:
        return "x_prediction"
    return "v_prediction"

def _image_tensor(obs, model):
    """Exact RobotWin deployment preprocessing, kept local to avoid a policy instance."""
    from PIL import Image
    cameras = obs["observation"]
    def resize(rgb, wh):
        return np.asarray(Image.fromarray(rgb.astype(np.uint8), "RGB").resize(wh, Image.BILINEAR), dtype=np.uint8)
    head = resize(cameras["head_camera"]["rgb"], (320, 256))
    left = resize(cameras["left_camera"]["rgb"], (160, 128))
    right = resize(cameras["right_camera"]["rgb"], (160, 128))
    mosaic = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
    image = torch.from_numpy(mosaic).permute(2, 0, 1).unsqueeze(0).to(model.device, model.torch_dtype)
    return image * (2.0 / 255.0) - 1.0

def _denormalise_action(normalizer, action: torch.Tensor) -> np.ndarray:
    """Return [T,D] for both public samplers and internal batched heads."""
    action = action.float().cpu()
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3 or action.shape[0] != 1:
        raise ValueError(f"Expected FastWAM action [T,D] or [1,T,D], got {tuple(action.shape)}")
    return normalizer.backward(action).numpy()[0]

@torch.no_grad()
def _action_cache(model, image, proprio, horizon, prompt):
    """Copied from FastWAM.infer_action up to (but not including) the ODE loop."""
    image=image.to(device=model.device,dtype=model.torch_dtype)
    first=model._encode_input_image_latents_tensor(input_image=image,tiled=False)
    context,mask=model.encode_prompt(prompt)
    context,mask=model._append_proprio_to_context(context=context,context_mask=mask,proprio=proprio)
    pre=model.video_expert.pre_dit(x=first,timestep=torch.zeros((1,),device=model.device,dtype=first.dtype),context=context,context_mask=mask,action=None,fuse_vae_embedding_in_latents=bool(getattr(model.video_expert,"fuse_vae_embedding_in_latents",False)))
    video_len=int(pre["tokens"].shape[1])
    attn=model._build_mot_attention_mask(video_seq_len=video_len,action_seq_len=horizon,video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]),device=pre["tokens"].device)
    cache=model.mot.prefill_video_cache(video_tokens=pre["tokens"],video_freqs=pre["freqs"],video_t_mod=pre["t_mod"],video_context_payload={"context":pre["context"],"mask":pre["context_mask"]},video_attention_mask=attn[:video_len,:video_len])
    return context,mask,cache,attn,video_len

def run(a):
    if not torch.cuda.is_available(): raise RuntimeError("FastWAM probe requires CUDA")
    sys.path[:0]=[str(FAST),str(FAST/"src")]
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    raw,rows=load_manifest_actions(a.manifest); specs=_specs(a.checkpoint); cfg=_cfg(a.task)
    outputs=[]
    for name,path in specs:
        print(f"[fastwam] loading checkpoint {name}: {path}", flush=True)
        model=instantiate(cfg.model,model_dtype=torch.bfloat16,device=a.device)
        model.load_checkpoint(str(path)); model=model.to(a.device).eval()
        # The checkpoint stores weights but this runtime choice lives in config.yaml.
        # Without this correction x-pred runs would silently take v-pred solver steps.
        model.action_prediction_type=_prediction_type(name,path)
        if a.kind == "nfe" and a.mode == "joint" and not a.decode_video:
            # infer_joint always decodes a video before returning.  The probe
            # only consumes its action, so skip this output-only VAE decode.
            model._decode_latents = lambda latents, tiled=False: None
            print("[fastwam] joint mode: final video decode disabled", flush=True)
        processor=instantiate(cfg.data.val.processor).eval(); processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(a.stats)))
        per_sample=[]
        selected_rows, selected_raw = rows[:a.max_samples], raw[:a.max_samples]
        total = len(selected_rows)
        for sample_index, (row,target_raw) in enumerate(zip(selected_rows,selected_raw), start=1):
            if sample_index == 1 or sample_index % a.progress_every == 0 or sample_index == total:
                print(f"[fastwam] {name}: anchor {sample_index}/{total}", flush=True)
            obs,state=_observation(row)
            image=_image_tensor(obs,model)
            meta=processor.shape_meta["state"][0]; key=meta["key"]
            batch={"state":{key:torch.as_tensor(state).unsqueeze(0)}}
            proprio=processor.normalizer.forward(processor.action_state_transform(batch))["state"][key]
            action_meta=processor.shape_meta["action"][0]; action_key=action_meta["key"]
            action_norm=processor.normalizer.normalizers["action"][action_key]
            # Normalizer statistics are CPU tensors; normalize there before moving to GPU.
            target_norm=action_norm.forward(torch.as_tensor(target_raw).unsqueeze(0)).to(model.device,model.torch_dtype)
            prompt=DEFAULT_PROMPT.format(task=str(row["instruction"]))
            grid=[]
            for value in (SIGMA_GRID if a.kind=="recovery" else NFE_GRID):
                seeds=[]
                for seed in range(a.seed,a.seed+a.noise_seeds):
                    if a.kind=="nfe":
                        if a.mode=="joint":
                            pred=model.infer_joint(prompt=prompt,input_image=image,num_video_frames=a.num_video_frames,action_horizon=raw.shape[1],proprio=proprio,num_inference_steps=int(value),seed=seed,rand_device="cpu",tiled=False,test_action_with_infer_action=False)["action"]
                        else:
                            pred=model.infer_action(prompt=prompt,input_image=image,action_horizon=raw.shape[1],proprio=proprio,num_inference_steps=int(value),seed=seed,rand_device="cpu",tiled=False)["action"]
                    else:
                        sigma=float(value); gen=torch.Generator(device=model.device).manual_seed(seed)
                        eps=torch.randn(target_norm.shape,generator=gen,device=model.device,dtype=model.torch_dtype)
                        z=((1-sigma)*target_norm+sigma*eps).to(model.torch_dtype)
                        context,mask,cache,attn,video_len=_action_cache(model,image,proprio,raw.shape[1],prompt)
                        time=torch.full((1,),sigma*model.infer_action_scheduler.num_train_timesteps,device=model.device,dtype=model.torch_dtype)
                        pred_head=model._predict_action_noise_with_cache(latents_action=z,timestep_action=time,context=context,context_mask=mask,video_kv_cache=cache,attention_mask=attn,video_seq_len=video_len)
                        pred=z-sigma*pred_head if model.action_prediction_type=="v_prediction" else pred_head
                    # Public `infer_*` returns [T,D]; the internal recovery
                    # head returns [1,T,D].  The archive always stores [T,D].
                    seeds.append(_denormalise_action(action_norm, pred))
                grid.append(np.stack(seeds))
            per_sample.append(np.stack(grid))
        outputs.append(np.stack(per_sample,axis=2))
        partial = Path(a.output).with_name(Path(a.output).stem + f".{name}.npz")
        key = "endpoint" if a.kind == "recovery" else "sample"
        np.savez_compressed(partial, **{key: np.stack(outputs[-1:]), "target": raw[:a.max_samples],
                                       ("sigma" if a.kind == "recovery" else "nfe"): (SIGMA_GRID if a.kind == "recovery" else NFE_GRID),
                                       "model": np.asarray([name])})
        print(f"[fastwam] wrote intermediate {partial}", flush=True)
        del model; torch.cuda.empty_cache()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    if a.kind=="recovery": np.savez_compressed(out,endpoint=np.stack(outputs),target=raw[:a.max_samples],sigma=SIGMA_GRID,model=np.asarray([n for n,_ in specs]))
    else: np.savez_compressed(out,sample=np.stack(outputs),target=raw[:a.max_samples],nfe=NFE_GRID,model=np.asarray([n for n,_ in specs]))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--kind",choices=("recovery","nfe"),default="nfe"); p.add_argument("--mode",choices=("action_only","joint"),default="action_only"); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--checkpoint",action="append",required=True); p.add_argument("--stats",type=Path,default=FAST/"data/robotwin2.0/dataset_stats.json"); p.add_argument("--task",default="robotwin_uncond_3cam_384_1e-4_clean50"); p.add_argument("--num-video-frames",type=int,default=9); p.add_argument("--output",type=Path,required=True); p.add_argument("--max-samples",type=int,default=2000); p.add_argument("--noise-seeds",type=int,default=8); p.add_argument("--seed",type=int,default=20260918); p.add_argument("--device",default="cuda"); p.add_argument("--progress-every",type=int,default=10); p.add_argument("--decode-video",action="store_true",help="Decode joint video output; off by default for action-only probes."); print(run(p.parse_args()))
if __name__=="__main__": main()
