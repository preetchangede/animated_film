"""Composite the boy's H3 eye performance behind the approved dry-cup plate."""
from pathlib import Path
import sys,subprocess
import numpy as np
from scipy.interpolate import PchipInterpolator
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image,ImageOps
here=Path(__file__).resolve().parent;r=here.parent
sys.path.insert(0,str(r/'qc-lib'))
import cv2 as cv
cv.setNumThreads(3);W,H=1920,1080;dim=(896,504)
plate=np.array(ImageOps.fit(Image.open(r/'guide/frames/08.png').convert('RGB'),(W,H),method=Image.Resampling.LANCZOS))[:,:,::-1].copy()
src=r/'motion-review/video_08_t01.mp4';cap=cv.VideoCapture(str(src));count=round(cap.get(cv.CAP_PROP_FRAME_COUNT));sw=cap.get(cv.CAP_PROP_FRAME_WIDTH);sh=cap.get(cv.CAP_PROP_FRAME_HEIGHT)
sift=cv.SIFT_create(nfeatures=2300)
def kp(f):return sift.detectAndCompute(cv.cvtColor(cv.resize(f,dim),cv.COLOR_BGR2GRAY),None)
base,ds=kp(plate);keys=[];trans=[]
for i in sorted(set(list(range(0,count,4))+[count-1])):
 cap.set(cv.CAP_PROP_POS_FRAMES,i);_,f=cap.read();pts,desc=kp(f)
 raw=cv.BFMatcher().knnMatch(desc,ds,k=2);good=[a for pair in raw if len(pair)==2 for a,b in [pair] if a.distance<.83*b.distance]
 srcxy=np.float32([pts[a.queryIdx].pt for a in good]);dstxy=np.float32([base[a.trainIdx].pt for a in good])
 M,inside=cv.estimateAffinePartial2D(srcxy,dstxy,method=cv.RANSAC,ransacReprojThreshold=2.)
 if M is None or inside.sum()<8:raise ValueError(f'Unstable cup tracking at frame {i}')
 keys.append(i);trans.append(M)
move=PchipInterpolator(keys,np.stack(trans),axis=0)
# The face is behind the near wall of the cup. The matte follows the upper edge and table.
polygon=np.array([(0,0),(375,0),(375,52),(310,56),(274,66),(237,85),(209,111),(193,132),(180,161),(169,163),(134,176),(107,179),(86,188),(51,215),(23,243),(0,252)],np.float32)
mask=np.zeros((H,W),np.uint8);cv.fillPoly(mask,[np.int32(polygon*[W/700,H/394])],255);alpha=cv.GaussianBlur(mask,(0,0),3)/255
cap.set(cv.CAP_PROP_POS_FRAMES,0)
proc=subprocess.Popen([get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','24','-i','-','-i',str(r/'render-review/video_01_t01.mp4'),'-map','0:v','-map','1:a','-af','lowpass=f=1700,volume=-20dB','-t',str(count/24),'-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','256k','-movflags','+faststart',str(here/'08-dry-cup.mp4')],stdin=subprocess.PIPE)
for i in range(count):
 _,f=cap.read()
 mat=np.vstack([move(i),[0,0,1]])
 mat=np.diag([W/dim[0],H/dim[1],1])@mat@np.diag([dim[0]/sw,dim[1]/sh,1])
 warped=cv.warpAffine(f,mat[:2],(W,H),flags=cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE)
 film=warped*alpha[:,:,None]+plate*(1-alpha[:,:,None]);proc.stdin.write(film.astype(np.uint8).tobytes())
proc.stdin.close()
if proc.wait():raise RuntimeError('Failed dry-cup encode')
