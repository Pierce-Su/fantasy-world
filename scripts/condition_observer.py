"""Isolated observer: real production return_condition/ImageEmbedderVAE; no denoise.
Only image-related units retained; text/noise/camera preparation intentionally omitted
because they do not produce y. DiT flags instantiated on meta, no DiT weights loaded.
"""
from pathlib import Path
import os,sys,json,time,hashlib,types,ast
import numpy as np
import torch
from PIL import Image
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('mode',choices=['fixture','measure'])
_parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
_parser.add_argument('--output-root',type=Path,required=True)
_parser.add_argument('--start-image',type=Path)
_parser.add_argument('--end-image',type=Path)
_parser.add_argument('--loader-log',type=Path)
_parser.add_argument('--vae-path',type=Path)
_parser.add_argument('--prompt-file',type=Path)
_parser.add_argument('--allow-vae',action='store_true',help='explicit permission for CONDITION_ONLY GPU VAE, never denoise')
ARGS=_parser.parse_args()
if ARGS.mode=='measure' and (not ARGS.allow_vae or any(getattr(ARGS,k) is None for k in ['start_image','end_image','loader_log','vae_path','prompt_file'])):
 _parser.error('measure requires --allow-vae and all explicit input/model paths')
repo=ARGS.repo.resolve();sys.path.insert(0,str(repo))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
if ARGS.mode=='measure' and not os.environ.get('HF_HOME'):
 _parser.error('set HF_HOME explicitly; no implicit cache/download policy')
from FantasyWorld.diffsynth_wan22.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_ImageEmbedderVAE
from FantasyWorld.diffsynth_wan22.models.wan_video_dit import WanModel
R=ARGS.output_root.resolve();R.mkdir(parents=True,exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def summary(t):
 t=t.detach().contiguous().cpu();f=t.float()
 return dict(shape=list(t.shape),dtype=str(t.dtype),finite=bool(torch.isfinite(f).all()),min=float(f.min()),max=float(f.max()),mean=float(f.mean()),absmax=float(f.abs().max()),sha256=sha(t.view(torch.uint8).numpy().tobytes()))
def observer(original,records):
 def wrapped(videos,*args,**kwargs):
  records.append({'vae_input':summary(videos[0]),'temporal':[summary(videos[0][:,i]) for i in range(videos[0].shape[1])], 'kwargs':{k:str(v) for k,v in kwargs.items()}})
  return original(videos,*args,**kwargs)
 return wrapped

def flags():
 # Saved loader kwargs are source evidence, not captured production DiT objects.
 log=ARGS.loader_log.read_text()
 rows=[]
 for line in log.splitlines():
  if 'This model is initialized with extra kwargs:' in line and 'in_dim' in line:
   kwargs=ast.literal_eval(line.split(':',1)[1].strip())
   with torch.device('meta'): m=WanModel(**kwargs)
   rows.append({'provenance':'real meta constructor + historical loader kwargs, NOT loaded DiT runtime', 'kwargs':kwargs,'flags':{k:getattr(m,k) for k in ['require_vae_embedding','require_clip_embedding','fuse_vae_embedding_in_latents','in_dim']}})
 assert len(rows)==2,len(rows)
 return rows

def fixture():
 pipe=WanVideoPipeline(device='cpu',torch_dtype=torch.float32)
 pipe.dit=types.SimpleNamespace(require_vae_embedding=True)
 class Fake:
  def encode(self,videos,**kw):
   v=videos[0];return torch.stack([v[:1,::4,::8,::8].repeat(16,1,1,1)])
 pipe.vae=Fake();unit=WanVideoUnit_ImageEmbedderVAE()
 start=Image.fromarray(np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3));end=Image.new('RGB',(16,16),(90,120,250))
 checks=[]
 for e in [None,end]:
  args=(pipe,start,e,9,16,16,True,(30,52),(15,26))
  bare=unit.process(*args)['y']; records=[];original=pipe.vae.encode;pipe.vae.encode=observer(original,records)
  observed=unit.process(*args)['y']; pipe.vae.encode=original
  assert torch.equal(bare,observed);assert len(records)==1
  checks.append(summary(observed))
 assert checks[0]['sha256']!=checks[1]['sha256']
 # Malformed/NaN summaries must not masquerade as finite data.
 assert not summary(torch.tensor([float('nan')]))['finite']
 out={'real_torch':torch.__version__,'device':'cpu','equal_before_after_hook':[True,True],'end_changes_fixture_y':True,'nan_detected':True,'checks':checks}
 (R/'fixture-result.json').write_text(json.dumps(out,indent=2)); print('FIXTURE PASS',flush=True)

@torch.no_grad()
def measure():
 fixture()
 paths={'start':ARGS.start_image,'end':ARGS.end_image}
 imgs={k:Image.open(p) for k,p in paths.items()};imgs['start']=imgs['start'].convert('RGB')
 identities={k:{'path':str(paths[k]),'file_sha256':sha(paths[k].read_bytes()),'mode':im.mode,'size':im.size,'rgb_sha256':sha(np.asarray(im).tobytes()),'resized_rgb_sha256':sha(np.asarray(im.resize((832,480))).tobytes())} for k,im in imgs.items()}
 ff=flags();assert ff[0]['flags']==ff[1]['flags']
 vaepath=ARGS.vae_path
 pipe=WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda',model_configs=[ModelConfig(path=str(vaepath))],tokenizer_config=None)
 pipe.dit=types.SimpleNamespace(**ff[0]['flags'])
 names=['WanVideoUnit_ShapeChecker','WanVideoUnit_ImageEmbedderVAE','WanVideoUnit_ImageEmbedderCLIP','WanVideoUnit_ImageEmbedderFused']
 pipe.units=[u for u in pipe.units if type(u).__name__ in names]
 assert len(pipe.units)==4
 original=pipe.vae.encode; records=[]; pipe.vae.encode=observer(original,records)
 ys={};runs={};prompt=ARGS.prompt_file.read_text().rstrip('\n')
 for label,end in [('without-end-image',None),('with-original-end-image',imgs['end']),('without-end-image-repeat',None)]:
  print('CONDITION_ONLY START '+label,flush=True);torch.manual_seed(1027);torch.cuda.manual_seed_all(1027)
  start=time.monotonic();shared,pos,neg=pipe(prompt=prompt,negative_prompt='',seed=1027,tiled=True,input_image=imgs['start'],end_image=end,height=480,width=832,num_frames=81,return_condition=True)
  y=shared['y'].cpu();ys[label]=y
  mask=y[0,:4];run={'y':summary(y),'mask_channels_by_group':mask[:,:,0,0].tolist(),'mask_spatially_constant':bool(torch.all(mask==mask[:,:,:1,:1])), 'normalized_images':{'start':summary(pipe.preprocess_image(imgs['start'].resize((832,480))))},'capture':records[-1],'elapsed_s':time.monotonic()-start}
  if end is not None:run['normalized_images']['end']=summary(pipe.preprocess_image(end.resize((832,480))))
  assert y.shape==(1,20,21,60,104) and run['y']['finite']
  out=R/'01_endpoint_condition'/label;out.mkdir(parents=True,exist_ok=True);(out/'measurements.json').write_text(json.dumps(run,indent=2));runs[label]=run
  print('CONDITION_ONLY DONE '+label+' '+str(run['elapsed_s']),flush=True)
 a=ys['without-end-image'];b=ys['with-original-end-image'];d=(b-a).float()
 assert torch.equal(a,ys['without-end-image-repeat']),'Real VAE OFF repeat nondeterministic'
 assert a[0,:4,-1,0,0].tolist()==[0,0,0,0] and b[0,:4,-1,0,0].tolist()==[0,0,0,1]
 assert not torch.equal(a,b) and torch.isfinite(d).all()
 diffs=[{'group':i,'all_channels':summary(d[:,:,i]),'latent_only':summary(d[:,4:,i])} for i in range(21)]
 result={'status':'CONDITION_ONLY_MEASURED_NO_DENOISE','scope':'production __call__(return_condition=True), unit_runner, ShapeChecker, ImageEmbedderVAE/CLIP/Fused; excluded unrelated text/noise/camera units, NO text/DiT weights, NO denoise or custom HIGH/LOW forward','images':identities,'DiT_flags':ff,'units_executed':names,'vae_path':str(vaepath),'settings':{'dtype':'bfloat16','tiled':True,'tile_size':[30,52],'tile_stride':[15,26],'seed_each_call':1027,'execution_order':list(runs)},'OFF_repeat_byte_equal':True,'runs':runs,'y_difference_per_group':diffs,'y_difference':summary(d),'all_finite':True,'end_mask_changed':True,'end_y_changed':True}
 (R/'01_endpoint_condition/measurements.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print('CONDITION_ONLY ALL PASS',flush=True)
if __name__=='__main__':
 if ARGS.mode=='fixture':fixture()
 elif ARGS.mode=='measure':measure()
