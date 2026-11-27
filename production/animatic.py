"""Cut the 15 opening poses against an original temporary Foley and whistle track."""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

RATE = 32000
W, H = 1280, 720
FPS = 24


def guide_sound(duration, output):
    rng = np.random.default_rng(53307)
    total = duration * RATE
    timeline = np.arange(total) / RATE
    mix = np.zeros((total, 2))

    def noise(size, low, high):
        raw = rng.normal(size=size)
        signal = sosfilt(butter(2, [low, high], btype='bandpass', fs=RATE, output='sos'), raw)
        return signal / max(np.std(signal), 0.001)

    # Keep a single uninterrupted bed across every cut. The roof insert and the
    # held beat have more distance; the ceramic strike gets air around its tail.
    room = np.interp(timeline, [0, 6, 8, 38, 51, 62, 68, 71, 83, 88, 94, 104, 117, 120],
                     [.92, 1, .68, .69, .60, .53, .43, .58, .53, .33, .43, .51, .64, .60])
    centre = noise(total, 360, 3500)
    surf = noise(total, 70, 350)
    for ch in range(2):
        swell = .57 + .34*np.sin(timeline*.55 + .7*np.sin(timeline*.21))
        mix[:, ch] = room*(.022*(.48*centre+.72*noise(total, 230, 5200)) + .022*surf*swell)

    def place(when, mono, pan):
        start = round(RATE * when)
        gains = [math.cos(math.pi*(pan+1)/4), math.sin(math.pi*(pan+1)/4)]
        for c, gain in enumerate(gains):
            end = min(total, start+len(mono))
            mix[start:end, c] += gain*mono[:end-start]
            for delay, level in ((.024+.006*c,.10),(.064-.009*c,.045)):
                k = start + int(RATE*delay)
                n = min(len(mono), total-k)
                if n>0:
                    mix[k:k+n,c] += gain*level*mono[:n]

    def ring(when, material, level=1.0):
        modes, decay, pan, gain = {
            'enamel': ([(395,1),(837,.48),(1432,.19)], .125, -.38,.24),
            'iron': ([(176,1),(302,.48),(613,.32),(931,.09)], .18, -.04,.26),
            'china': ([(1421,1),(3148,.32),(4157,.095)], .39,.30,.26),
            'wood': ([(103,1),(226,.54),(458,.24)], .049,.23,.17),
        }[material]
        t=np.arange(round(RATE*(decay*7)))/RATE
        note=np.zeros_like(t)
        shift=rng.uniform(.994,1.006)
        for i,(hz,a) in enumerate(modes):
            note += a*np.sin(2*np.pi*hz*shift*t)*np.exp(-t*(1+.28*i)/decay)
        note += .17*noise(t.size,700,5200)*np.exp(-t/.008)
        place(when, (1-np.exp(-t/.002))*note*gain*level, pan)

    def whistle(when, pitch, length, level):
        t=np.arange(round(length*RATE))/RATE
        breath=noise(t.size, 1200, 4800)
        f=pitch*(1-.029*np.exp(-t/.060)+.0032*np.sin(2*np.pi*4.5*t+.25))
        phase=2*np.pi*np.cumsum(f)/RATE
        env=(1-np.exp(-t/.045))*(1-np.exp(-(length-t)/.074))
        place(when, level*.16*env*(np.sin(phase)+.057*np.sin(phase*2)+.028*breath),.16)

    def phrase(when, notes, level=.88):
        for offset,pitch,length in notes:
            whistle(when+offset,pitch,length,level)

    tune=[(0,523.25,.39),(.61,587.33,.36),(1.40,698.46,.46),(2.12,587.33,.77)]
    phrase(10.10,tune,.68)
    for when,kind,level in [
            (15.9,'wood',.22),(17.8,'enamel',.80),(20.3,'enamel',.66),
            (24.2,'wood',.25),(26.0,'iron',.85),(28.55,'iron',.62),
            (33.15,'wood',.41),(35.12,'enamel',.66),(35.86,'iron',.64),
            (40.10,'enamel',.96),(40.80,'iron',.90),(44.2,'enamel',.56),(44.98,'iron',.58),
            (48.6,'enamel',.87),(49.37,'iron',.85),(53.60,'enamel',.54),(54.42,'iron',.72),
            (65.81,'wood',.37),(71.9,'enamel',.45),(72.68,'iron',.48),
            (90.90,'china',1),(107.75,'china',.76),
            (112.5,'wood',.39),(113.55,'enamel',.63),(114.24,'iron',.64),
            (115.03,'china',.60),(115.70,'wood',.79),(116.50,'wood',.59)]:
        ring(when,kind,level)
    phrase(41.10,tune[:2],.8)
    phrase(49.68,tune[:2],.73)
    phrase(94.28,tune,.80)
    phrase(103.35,tune[:2],.74)
    phrase(114.94,[tune[0],(.62,587.33,.41),(1.37,523.25,.73)],.59)

    def rub(when, length, pan, level):
        t=np.arange(round(length*RATE))/RATE
        env=np.sin(np.pi*t/length)**1.7
        place(when,level*noise(len(t),450,3200)*env*(.78+.22*np.cos(t*49)),pan)
    for args in [(8.52,.42,-.35,.020),(15.85,.44,-.34,.035),(24.50,.66,.08,.041),
                 (32.86,.22,.35,.026),(51.52,.34,-.21,.018),(65.67,.31,-.24,.028),
                 (70.57,.77,-.25,.050),(80.65,.47,-.15,.032),(83.00,.20,-.1,.018),
                 (111.95,.61,-.22,.042)]:
        rub(*args)

    # Leave a clean lead-in and a gentle tail that survives a player auto-loop.
    mix *= np.minimum(1,timeline/.65)[:,None]*np.clip((duration-timeline)/1.40,0,1)[:,None]
    mix *= .78/max(np.abs(mix).max(),.001)
    wavfile.write(output, RATE, (mix*32767).astype(np.int16))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('shots',type=Path)
    p.add_argument('frames',type=Path,help='Directory containing numbered 01.png through 15.png')
    p.add_argument('output',type=Path)
    args=p.parse_args()
    shots=json.loads(args.shots.read_text())
    output=args.output.resolve()
    scratch=output.parent
    guide=scratch/'timing-guide.wav'
    runtime=sum(s['seconds'] for s in shots)
    guide_sound(runtime,guide)
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg=get_ffmpeg_exe()
    # A 24 fps timebase gives exact cut points even when two panels have equal exposure.
    lines=['ffconcat version 1.0']
    for s in shots:
        n=s['shot']
        source=args.frames/f'{n:02d}.png'
        plate=scratch/f'plate-{n:02d}.png'
        with Image.open(source) as im:
            ImageOps.fit(im.convert('RGB'),(W,H),method=Image.Resampling.LANCZOS).save(plate)
        lines.extend([f"file '{plate.name}'",'option framerate 24',f"duration {s['seconds']}"])
    lines.extend([f"file '{plate.name}'",'option framerate 24'])
    edit=scratch/'timeline.ffconcat'
    edit.write_text('\n'.join(lines)+'\n')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','warning','-y','-safe','0','-f','concat',
                    '-i',str(edit),'-i',str(guide),'-map','0:v:0','-map','1:a:0','-vf','fps=24:round=up,tpad=stop_mode=clone:stop_duration=2',
                    '-t',str(runtime),'-c:v','libx264','-preset','fast','-tune','stillimage',
                    '-crf','17','-pix_fmt','yuv420p','-colorspace','bt709','-color_primaries','bt709',
                    '-color_trc','bt709','-c:a','aac','-b:a','192k','-movflags','+faststart',
                    '-metadata','title=The Missing Drip | Timing animatic',str(output)],check=True)

if __name__=='__main__':
    main()
