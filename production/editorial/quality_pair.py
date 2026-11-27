"""Compare two refinements after the exact 1080p master crop, including faces, hands and paper grain."""
from pathlib import Path
from PIL import Image,ImageDraw
import sys
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'qc-lib'))
import cv2 as cv
def frame(cap,t):
 cap.set(cv.CAP_PROP_POS_MSEC,t*1000);ok,f=cap.read();assert ok
 # Scale to 1920 wide, then crop centrally as the master will.
 h=round(f.shape[0]*1920/f.shape[1]);f=cv.resize(f,(1920,h),interpolation=cv.INTER_LANCZOS4)
 start=(h-1080)//2;return f[start:start+1080]
def ssim(a,b):
 a=cv.cvtColor(a,cv.COLOR_BGR2GRAY).astype(np.float32)/255
 b=cv.cvtColor(b,cv.COLOR_BGR2GRAY).astype(np.float32)/255
 blur=lambda x:cv.GaussianBlur(x,(11,11),1.5)
 ma,mb=blur(a),blur(b);va,vb=blur(a*a)-ma**2,blur(b*b)-mb**2;cov=blur(a*b)-ma*mb
 return np.mean(((2*ma*mb+.0001)*(2*cov+.0009))/((ma*ma+mb*mb+.0001)*(va+vb+.0009)))
a,b=[cv.VideoCapture(x) for x in sys.argv[1:3]];scores=[]
sheet=Image.new('RGB',(1000,4*270),(30,30,30));draw=ImageDraw.Draw(sheet)
for i,(t,x,y) in enumerate([(.42,380,115),(2.5,1230,169),(4.7,1200,470),(6.9,1420,650)]):
 A,B=frame(a,t),frame(b,t);roi=(slice(y,y+240),slice(x,x+490));scores.append((t,round(float(ssim(A,B)),4),round(float(ssim(A[roi],B[roi])),4)))
 for c,img in enumerate([A,B]):
  crop=Image.fromarray(cv.cvtColor(img[roi],cv.COLOR_BGR2RGB));sheet.paste(crop,(500*c,270*i));draw.text((500*c+2,270*i+243),f'{t}s  '+f'{int([a,b][c].get(cv.CAP_PROP_FRAME_WIDTH))} px source',fill='white')
sheet.save(root/'finish-comparison.jpg',quality=96)
print('Time / full-image SSIM / detail SSIM',scores)
