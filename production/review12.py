"""Register H3's macro camera move to the paper cup and inspect the moving drop."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'qc-lib'))
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
src=root/'render-review/video_12_t01.mp4'
capture=cv.VideoCapture(str(src))
print(cv.__version__,capture.isOpened(),capture.get(cv.CAP_PROP_FRAME_COUNT),flush=True)
width,height=1008,576
cv.setNumThreads(2)
sift=cv.SIFT_create(nfeatures=1700)
bf=cv.BFMatcher(cv.NORM_L2)
def features(frame):
    grey=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    return sift.detectAndCompute(grey,None)
def match(frame,base):
    kp,des=features(frame)
    raw=bf.knnMatch(des,base[1],k=2)
    good=[m for pair in raw if len(pair)==2 for m,n in [pair] if m.distance<.8*n.distance]
    a=np.float32([kp[m.queryIdx].pt for m in good])
    b=np.float32([base[0][m.trainIdx].pt for m in good])
    transform,ok=cv.findHomography(a,b,cv.RANSAC,2)
    return transform,int(ok.sum())
def run():
    images=[];counts=[]
    for t in [0,1,2,3,4,5,6,7,8,9]:
        capture.set(cv.CAP_PROP_POS_MSEC,t*1000)
        ok,fr=capture.read()
        if not ok: continue
        f=cv.resize(fr,(width,height),interpolation=cv.INTER_AREA)
        if not images:
            base=features(f)
            cv.imwrite(str(root/'macro-base.jpg'),f)
        transform,n=match(f,base)
        warped=cv.warpPerspective(f,transform,(width,height))
        counts.append(n)
        small=cv.resize(warped,(504,288))
        images.append(Image.fromarray(cv.cvtColor(small,cv.COLOR_BGR2RGB)))
    sheet=Image.new('RGB',(1008,5*304))
    draw=ImageDraw.Draw(sheet)
    for i,im in enumerate(images):
        x=i%2*504;y=i//2*304
        sheet.paste(im,(x,y));draw.text((x+3,y+289),f'{i}s · {counts[i]} anchors')
    sheet.save(root/'macro-alignment.jpg')
    print(counts,flush=True)
if __name__=='__main__': run()
