"""Portable version of historical60/90 comparison; originals are never modified."""
from pathlib import Path
import subprocess,json,hashlib,io
from PIL import Image,ImageDraw,ImageFont
import argparse
p=argparse.ArgumentParser(description='Compare existing 81-frame832x480@16fps60/90 clips. No inference. Outputs are presentation-only reencodes.')
p.add_argument('--clip60',type=Path,required=True)
p.add_argument('--clip90',type=Path,required=True)
p.add_argument('--output-root',type=Path,required=True)
p.add_argument('--font',type=Path)
a=p.parse_args();D=a.output_root.resolve();D.mkdir(parents=True,exist_ok=False)
A=a.clip60.resolve();B=a.clip90.resolve()
font=lambda n:ImageFont.truetype(str(a.font),n) if a.font else ImageFont.load_default(size=n)
probes=[];thumbs={};selected={}
for label,path in [('60',A),('90',B)]:
 p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(path)],capture_output=True,text=True,check=True);meta=json.loads(p.stdout)['streams'][0]
 assert meta['nb_read_frames']=='81' and meta['width']==832 and meta['height']==480 and meta['r_frame_rate']=='16/1'
 p=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True);assert p.returncode==0 and not p.stderr
 probes.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'ffprobe':meta,'full_decode_exit':p.returncode,'stderr':p.stderr})
 proc=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 frames=[]
 for i in range(81):
  raw=proc.stdout.read(832*480*3);assert len(raw)==832*480*3
  im=Image.frombytes('RGB',(832,480),raw)
  if i in [0,20,40,60,80]:selected[label,i]=im.copy()
  frames.append(im.resize((312,180)))
 assert proc.stdout.read(1)==b'';assert proc.wait()==0;assert not proc.stderr.read();proc.stdout.close();proc.stderr.close();thumbs[label]=frames
sheet=Image.new('RGB',(1320,2075),'#111a28');dr=ImageDraw.Draw(sheet)
dr.text((24,15),'Same frame / same time: requested CCW 60 vs 90 (not measured video poses)',font=font(24),fill='white')
dr.text((24,52),'60 degrees test - NEW',font=font(23),fill='#79e4d0');dr.text((675,52),'90 degrees reference - REUSED',font=font(23),fill='#ffbf86')
for row,i in enumerate([0,20,40,60,80]):
 y=95+row*390;dr.text((24,y),f'f{i} / {i/16:.2f}s | commanded angles {60*i/80:g} vs {90*i/80:g} deg',font=font(22),fill='white')
 for x,label in [(24,'60'),(675,'90')]:sheet.paste(selected[label,i].resize((624,360)),(x,y+30))
sheet.save(D/'orbit-060-vs-090_timeline.jpg',quality=94)
review=D/'all-frame-review';review.mkdir(exist_ok=True)
for page in range(9):
 canvas=Image.new('RGB',(1920,700),'#111a28');draw=ImageDraw.Draw(canvas)
 draw.text((15,8),f'All-frame review {page+1}/9 | each pair: 60 LEFT, 90 RIGHT | decoded MP4',font=font(23),fill='white')
 for j in range(9):
  i=page*9+j;x=(j%3)*640+5;y=(j//3)*215+45
  draw.text((x,y),f'f{i}  t={i/16:.4f}s',font=font(18),fill='white')
  canvas.paste(thumbs['60'][i],(x,y+25));canvas.paste(thumbs['90'][i],(x+317,y+25))
 canvas.save(review/f'frames-{page*9:02d}-{page*9+8:02d}.jpg',quality=93)
side=D/'orbit-060-vs-090_side-by-side.mp4'
filter="[0:v]pad=iw:ih+40:0:40:color=0x111a28,drawtext=text='CCW 60 deg TEST (NEW)':x=15:y=10:fontsize=22:fontcolor=white[a];[1:v]pad=iw:ih+40:0:40:color=0x111a28,drawtext=text='CCW 90 deg REFERENCE (REUSED)':x=15:y=10:fontsize=22:fontcolor=white[b];[a][b]hstack=inputs=2[v]"
cmd=['ffmpeg','-v','error','-n','-i',str(A),'-i',str(B),'-filter_complex',filter,'-map','[v]','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(side)]
p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr
p=subprocess.run(['ffmpeg','-v','error','-i',str(side),'-f','null','-'],capture_output=True,text=True);assert p.returncode==0 and not p.stderr
assert side.stat().st_size<20*1024*1024
(D/'validation.json').write_text(json.dumps({'videos':probes,'side_by_side':{'sha256':hashlib.sha256(side.read_bytes()).hexdigest(),'bytes':side.stat().st_size,'full_decode_exit':p.returncode,'command':cmd},'review_sheets':9,'same_frame_samples':[0,20,40,60,80]},indent=2))
html='''<!doctype html><meta charset="utf-8"><title>Orbit comparison playback</title><style>body{background:#111a28;color:white;font:20px sans-serif}video{width:100%;max-width:1664px}</style><h1>60 test (left) / reused 90 reference (right)</h1><video id="video" controls muted src="orbit-060-vs-090_side-by-side.mp4"></video><pre id="status"></pre><script>window.evidence={frames:[],ended:false,error:null};let v=document.querySelector('video');function frame(now,m){evidence.frames.push(m.mediaTime);v.requestVideoFrameCallback(frame)}v.requestVideoFrameCallback(frame);v.onended=()=>{evidence.ended=true;evidence.duration=v.duration;evidence.quality=v.getVideoPlaybackQuality();document.querySelector('#status').textContent=JSON.stringify(evidence)};v.onerror=()=>{evidence.error=v.error?.message};</script>'''
(D/'playback.html').write_text(html)
print(json.dumps(probes,indent=2));print('side bytes',side.stat().st_size)
