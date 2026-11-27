"""Preserve H3's whistle and cup timbre for recurring cues across independently generated shots."""
from pathlib import Path
import subprocess
import numpy as np
from scipy.io.wavfile import write
from scipy.ndimage import median_filter
from scipy.signal import stft,istft
from imageio_ffmpeg import get_ffmpeg_exe
root=Path(__file__).resolve().parent
sr=48000
def read(path,start,duration):
    cmd=[get_ffmpeg_exe(),'-v','error','-ss',str(start),'-i',str(path),'-t',str(duration),'-vn','-ar',str(sr),'-ac','2','-f','f32le','-']
    return np.frombuffer(subprocess.run(cmd,capture_output=True,check=True).stdout,np.float32).reshape(-1,2)
def taper(a,attack=.02,release=.07):
    n=min(round(attack*sr),len(a));m=min(round(release*sr),len(a))
    a[:n]*=np.sin(np.linspace(0,np.pi/2,n))[:,None]**2
    a[-m:]*=np.cos(np.linspace(0,np.pi/2,m))[:,None]**2
    return a
def save(name,a):
    write(root/name,sr,(np.clip(a,-1,1)*32767).astype(np.int16))
def get_whistle(start,end):
    samples=read(root.parent/'render-review/video_14_t01.mp4',start,end-start)
    f,t,z=stft(samples.T,fs=sr,nperseg=2048,noverlap=1792)
    power=np.abs(z).mean(axis=0)
    band=(f>=850)&(f<2100)
    pitch=median_filter(f[band][np.argmax(power[band],axis=0)],size=9)
    envelope=np.max(power[band],axis=0)
    mask=np.zeros(power.shape)
    for h in range(1,6):
        mask=np.maximum(mask,np.exp(-.5*((f[:,None]-h*pitch[None,:])/(40+h*22))**2))
    # Retain a low level of the real breath surrounding the line.
    mask=np.maximum(mask,.04*np.clip((f[:,None]-650)/450,0,1))
    gate=np.clip(envelope/(np.quantile(envelope,.80)*.12+1e-7),0,1)
    _,wave=istft(z*mask*gate,fs=sr,nperseg=2048,noverlap=1792)
    return taper(wave.T[:len(samples)].copy())
save('granddad-phrase.wav',get_whistle(.05,4.12))
save('granddad-answer.wav',get_whistle(4.78,8.65))
save('cup-strike.wav',taper(read(root.parent/'render-review/video_12_t01.mp4',7.035,1.77).copy(),.008,.16))
