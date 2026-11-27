"""Assemble the reviewed H3 takes, including editorial handles and their native sound."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def ffmpeg_path():
    binary=shutil.which('ffmpeg')
    if binary:
        return binary
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def inspect(ffmpeg, path):
    # ffmpeg is also available in portable environments without a separate ffprobe.
    report=subprocess.run([ffmpeg,'-hide_banner','-i',str(path)],capture_output=True,text=True).stderr
    duration=re.search(r'Duration: (\d+):(\d+):([\d.]+)',report)
    if not duration or ': Video:' not in report or ': Audio:' not in report:
        raise ValueError(f'Missing native video or audio in {path.name}')
    h,m,s=duration.groups()
    return int(h)*3600+int(m)*60+float(s)


def master(shots, decisions, output, mix=None, title=None, credits=None):
    ffmpeg=ffmpeg_path()
    if [s['shot'] for s in shots]!=[s['shot'] for s in decisions]:
        raise ValueError('Every shot needs one matching, reviewed edit decision in story order')
    args=[ffmpeg,'-hide_banner','-loglevel','warning','-nostdin','-y','-filter_complex_threads','3']
    filters=[]
    total=sum(s['seconds'] for s in shots)
    count=len(shots)
    for i,(shot,d) in enumerate(zip(shots,decisions)):
        source=Path(d['file']).resolve(strict=True)
        start=float(d.get('in',0))
        overlap=float(d.get('sound_tail',.08)) if i<count-1 else 0
        level=float(d.get('audio_gain_db',0))
        if start<0 or not 0<=overlap<=.5:
            raise ValueError(f'Invalid cut or sound handle for {source.name}')
        seconds=float(shot['seconds'])
        if inspect(ffmpeg,source)+.02<start+seconds+overlap:
            raise ValueError(f'Too little source or audio handle in {source.name}')
        args+=['-threads','2','-ss',str(start),'-i',str(source)]
        filters.append(f'[{i}:v]trim=duration={seconds},setpts=PTS-STARTPTS,fps=24,'
                       'scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,'
                       f'crop=1920:1080,setsar=1,format=yuv420p[v{i}]')
        if not mix:
            filters.append(f'[{i}:a]aresample=48000,aformat=channel_layouts=stereo,'
                           f'atrim=duration={seconds+overlap},asetpts=PTS-STARTPTS,volume={level}dB[a{i}]')
    filters.append(''.join(f'[v{i}]' for i in range(count))+f'concat=n={count}:v=1:a=0[picture]')
    a='a0'
    idx=count
    if mix:
        args+=['-i',str(Path(mix).resolve(strict=True))]
        filters.append(f'[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=duration={total},asetpts=PTS-STARTPTS[aext]')
        a='aext'
        idx+=1
    else:
        for i in range(1,count):
            tail=float(decisions[i-1].get('sound_tail',.08))
            op=f'acrossfade=d={tail}:c1=qsin:c2=qsin' if tail>0 else 'concat=n=2:v=0:a=1'
            filters.append(f'[{a}][a{i}]{op}[ax{i}]')
            a=f'ax{i}'
    # Cards are edited as transparent full-coverage PNGs. The final gesture stays visible.
    last_picture='picture'
    for tag,path,start,end in [('title',title,1.0,6.25),('credits',credits,total-3.0,total-.2)]:
        if not path:
            continue
        args+=['-framerate','24','-loop','1','-i',str(Path(path).resolve(strict=True))]
        filters.append(f'[{idx}:v]format=rgba,fade=in:st={start}:d=0.55:alpha=1,'
                       f'fade=out:st={end-.45}:d=.45:alpha=1[{tag}alpha]'.replace('d=.45','d=0.45'))
        filters.append(f'[{last_picture}][{tag}alpha]overlay=shortest=1:enable=\'between(t,{start},{end})\'[{tag}out]')
        last_picture=tag+'out'
        idx+=1
    filters.append(f'[{last_picture}]fade=out:st={total-.6}:d=0.6[vout]')
    filters.append(f'[{a}]afade=in:st=0:d=0.12,afade=out:st={total-.75}:d=0.75[aout]')
    args+=['-filter_complex',';'.join(filters),'-map','[vout]','-map','[aout]','-t',str(total),
           '-c:v','libx264','-preset','slow','-crf','16','-profile:v','high','-pix_fmt','yuv420p',
           '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-c:a','aac',
           '-ar','48000','-ac','2','-b:a','320k','-movflags','+faststart','-metadata','title=The Missing Drip']
    output=output.resolve()
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        rendered=Path(tmp)/'master.mp4'
        subprocess.run(args+[str(rendered)],check=True)
        if abs(inspect(ffmpeg,rendered)-total)>.04:
            raise ValueError('Encoded duration diverged from picture lock')
        subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(rendered),'-map','0:v','-map','0:a','-f','null','-'],check=True)
        os.replace(rendered,output)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('shots',type=Path)
    p.add_argument('decisions',type=Path,help='Shot numbers, reviewed take paths, trim-in, optional sound_tail and audio_gain_db.')
    p.add_argument('output',type=Path)
    p.add_argument('--mix',type=Path,help='Final reviewed stereo mix with match-to-picture timing')
    p.add_argument('--title',type=Path)
    p.add_argument('--credits',type=Path)
    a=p.parse_args()
    master(json.loads(a.shots.read_text()),json.loads(a.decisions.read_text()),a.output,a.mix,a.title,a.credits)
