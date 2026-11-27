"""Retain H3's clean, unbroken performance after the exposure change, preserving 24 fps."""
from pathlib import Path
import sys,subprocess
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'qc-lib'))
import cv2 as cv
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from scipy.interpolate import PchipInterpolator
from scipy.io.wavfile import read,write
cv.setNumThreads(3);ff=get_ffmpeg_exe();sr=48000
src=root.parent/'render-review/video_14_t01.mp4'
cap=cv.VideoCapture(str(src));w,h=2016,1152
small=(504,288)
def grey(a):return cv.cvtColor(cv.resize(a,small,interpolation=cv.INTER_AREA),cv.COLOR_BGR2GRAY)
def audio():
    sound=np.zeros((int(sr*9.36),2),np.float32)
    def place(name,start,gain):
        fs,a=read(root/name); assert fs==sr
        sound[round(start*sr):round(start*sr)+len(a)]+=a.astype(np.float32)/32768*gain
    place('granddad-phrase.wav',.25,.9);place('cup-strike.wav',6.53,.15)
    write(root/'14-sound.wav',sr,(np.clip(sound,-1,1)*32767).astype(np.int16))
audio()
proc=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','24','-i','-','-i',str(root/'14-sound.wav'),'-c:v','libx264','-preset','fast','-crf','15','-pix_fmt','yuv420p','-c:a','aac','-b:a','256k','-shortest',str(root/'14-smile-v01.mp4')],stdin=subprocess.PIPE)
flow=cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
gy,gx=np.indices((h,w),dtype=np.float32)
positions=PchipInterpolator([0,1.4,4.2,6.35,7.25,9.30],[5.06,5.77,7.05,8.46,9.04,10.04])
cur=-2;cap.set(cv.CAP_PROP_POS_FRAMES,int(positions(0)*24))
for frame in range(224):
    a=float(positions(frame/24))*24;k=int(a);t=a-k
    if k!=cur:
        if cur<0:
            _,l=cap.read();l=cv.resize(l,(w,h),interpolation=cv.INTER_AREA)
        else:l=r
        ok,n=cap.read()
        r=cv.resize(n,(w,h),interpolation=cv.INTER_AREA) if ok else l
        v=flow.calc(grey(l),grey(r),None);back=flow.calc(grey(r),grey(l),None)
        v=cv.resize(v,(w,h))*4;back=cv.resize(back,(w,h))*4;cur=k
    left=cv.remap(l,gx-t*v[:,:,0],gy-t*v[:,:,1],cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE)
    right=cv.remap(r,gx-(1-t)*back[:,:,0],gy-(1-t)*back[:,:,1],cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE)
    proc.stdin.write(cv.addWeighted(left,1-t,right,t,0).tobytes())
proc.stdin.close()
if proc.wait():raise RuntimeError('Encoder failed')
