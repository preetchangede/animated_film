"""Anchor the late pan on the cottage walls, restoring the original framing around the listening boy."""
import argparse,sys,subprocess
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
parent=Path(__file__).resolve().parent
sys.path.insert(0,str(parent.parent/'qc-lib'))
import cv2 as cv
from imageio_ffmpeg import get_ffmpeg_exe
cv.setNumThreads(3)
def render(src,out):
    cap=cv.VideoCapture(str(src))
    total=round(cap.get(cv.CAP_PROP_FRAME_COUNT)); w,h=map(lambda k:round(cap.get(k)),[cv.CAP_PROP_FRAME_WIDTH,cv.CAP_PROP_FRAME_HEIGHT])
    output_w=min(2016,w);output_h=round(output_w*h/w)
    dim=(840,480);sift=cv.SIFT_create(nfeatures=1300)
    def features(img):return sift.detectAndCompute(cv.cvtColor(cv.resize(img,dim),cv.COLOR_BGR2GRAY),None)
    _,first=cap.read()
    # Both files decode the same motion. Solve the camera once in motion-proof space,
    # where diffusion refinement has not changed the tiny repeat texture on the staircase.
    proof=parent.parent/'motion-review'/src.name
    tracker=cv.VideoCapture(str(proof if proof.exists() else src))
    _,guide=tracker.read();base,kd=features(guide)
    steps=list(range(0,total,4));steps.append(total-1) if steps[-1]!=total-1 else None
    transforms=[]
    for i in steps:
        tracker.set(cv.CAP_PROP_POS_FRAMES,i);_,f=tracker.read();pts,desc=features(f)
        match=cv.BFMatcher().knnMatch(desc,kd,k=2);good=[m for pair in match if len(pair)==2 for m,n in [pair] if m.distance<.76*n.distance]
        source=np.float32([pts[m.queryIdx].pt for m in good]);target=np.float32([base[m.trainIdx].pt for m in good])
        stable=(target[:,0]<280)|((target[:,0]>650)&(target[:,1]>50))
        if stable.sum()>12:source,target=source[stable],target[stable]
        M,ok=cv.estimateAffinePartial2D(source,target,method=cv.RANSAC,ransacReprojThreshold=2.3)
        if M is None or ok.sum()<6:raise ValueError(f'Not enough stationary cottage features at {i}')
        transforms.append(M)
    interp=PchipInterpolator(steps,np.stack(transforms),axis=0);cx=output_w/dim[0]
    first=cv.resize(first,(output_w,output_h),interpolation=cv.INTER_AREA)
    mask=np.zeros((output_h,output_w),np.uint8)
    cv.ellipse(mask,(round(output_w*.57),round(output_h*.55)),(round(output_w*.33),round(output_h*.60)),0,0,360,255,-1)
    alpha=cv.GaussianBlur(mask,(0,0),output_w*.014).astype(np.float32)/255
    alpha*=np.clip((np.arange(output_w)/output_w-.335)/.045,0,1)[None,:]
    ff=get_ffmpeg_exe();p=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{output_w}x{output_h}','-r','24','-i','-','-i',str(src),'-map','0:v','-map','1:a','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    coverage=np.full((h,w),255,np.uint8)
    cap.set(cv.CAP_PROP_POS_FRAMES,0)
    for i in range(total):
        _,f=cap.read();M=interp(i).copy();M[:,:2]*=output_w/w;M[:,2]*=cx
        frame=cv.warpAffine(f,M,(output_w,output_h),flags=cv.INTER_CUBIC)
        valid=cv.warpAffine(coverage,M,(output_w,output_h),flags=cv.INTER_NEAREST)>0
        blend=alpha*np.minimum(1,cv.distanceTransform(valid.astype(np.uint8),cv.DIST_L2,3)/18)
        p.stdin.write((frame*blend[:,:,None]+first*(1-blend[:,:,None])).astype(np.uint8).tobytes())
    p.stdin.close()
    if p.wait():raise RuntimeError('Encode failed')
    print('Frame bounds',np.min(np.stack(transforms)[:,:,2],axis=0),np.max(np.stack(transforms)[:,:,2],axis=0))
if __name__=='__main__':render(Path(sys.argv[1]),Path(sys.argv[2]))
