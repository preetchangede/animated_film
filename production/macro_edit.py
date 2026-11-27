"""Reframe the H3 macro with the cup as a locked camera anchor; reuse its impact sound."""
from review12 import root,src,cv,np,width,height,features
from PIL import Image
from imageio_ffmpeg import get_ffmpeg_exe
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
import subprocess
from pathlib import Path
cv.setNumThreads(3)
outdir=root/'editorial';outdir.mkdir(exist_ok=True)
ff=get_ffmpeg_exe()
w,h=2016,1152
def low(f):return cv.resize(f,(width,height),interpolation=cv.INTER_AREA)
def calc(f,base):
    key,desc=features(f)
    pairs=cv.BFMatcher(cv.NORM_L2).knnMatch(desc,base[1],k=2)
    matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.77*b.distance]
    xy=np.float32([key[m.queryIdx].pt for m in matches]);ref=np.float32([base[0][m.trainIdx].pt for m in matches])
    matrix,inliers=cv.findHomography(xy,ref,cv.RANSAC,1.65)
    if matrix is None or inliers.sum()<10:raise ValueError('Registration could not establish cup geometry')
    return matrix/matrix[2,2]
def assemble():
 cap=cv.VideoCapture(str(src));ok,first=cap.read();cap.set(cv.CAP_PROP_POS_FRAMES,0)
 first=cv.resize(first,(w,h),interpolation=cv.INTER_AREA)
 path=outdir/'macro-transform.npy'
 if path.exists(): transforms=np.load(path)
 else:
    transforms=[]
    for i in range(round(cap.get(cv.CAP_PROP_FRAME_COUNT))):
        ok,f=cap.read()
        if not ok:break
        small=low(f)
        if i==0:
            base=features(small);ref=np.eye(3)
        local=calc(small,base);total=ref @ local;total/=total[2,2]
        transforms.append(total)
        if i%16==0:
            base=features(small);ref=total
    transforms=savgol_filter(np.array(transforms),13,2,axis=0)
    np.save(path,transforms)
 cap.release()
 # Paint out the first-frame droplet underneath the spoon using the adjacent paper and the curved far rim.
 yy,xx=np.indices((h,w),dtype=np.float32); xn=xx/2;yn=yy/2
 mask=np.zeros((h,w),np.uint8)
 cv.fillPoly(mask,[2*np.array([[467,162],[527,158],[532,164],[528,187],[521,198],[510,204],[486,207],[475,198],[469,188]],np.int32)],255)
 mask=cv.dilate(mask,np.ones((7,7),np.uint8));mask=cv.GaussianBlur(mask,(0,0),1.5)
 # Continue the background hue and the rear lip from its closest intact section.
 left=cv.remap(first,xx-175,(yy+3.).astype(np.float32),cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE)
 right=cv.remap(first,xx+175,(yy+3.).astype(np.float32),cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE)
 blend=np.clip((xn-461)/75,0,1)[:,:,None]
 shifted=left*(1-blend)+right*blend
 clean=(first*(1-mask[:,:,None]/255)+shifted*(mask[:,:,None]/255)).astype(np.uint8)
 # Keep the spoon and the outside silhouette still. Blend only H3's moving bead and interior ripple.
 mask_near=np.zeros((h,w),np.uint8)
 cv.ellipse(mask_near,(998,368),(110,104),0,0,360,255,-1)
 cv.ellipse(mask_near,(990,529),(145,160),0,0,360,255,-1)
 cv.ellipse(mask_near,(986,684),(222,74),0,0,360,255,-1)
 mask_near=cv.GaussianBlur(mask_near,(0,0),13)
 # The native audio impact is at 7.12. Map it to 4.90 so the empty beat pays off.
 map_to_source=PchipInterpolator([0,.4,2.9,4.3,4.64,4.90,5.36,6.40,8.30],[0,.16,1.20,3.5,5.58,7.12,7.63,8.42,9.31])
 cap=cv.VideoCapture(str(src));source_idx=-1;cur=None
 out=outdir/'12-macro-v02.mp4'
 cmd=[ff,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','24','-i','-','-i',str(src),
      '-filter_complex','[1:a]atrim=start=2.22,asetpts=PTS-STARTPTS,volume=-8dB,apad[a]',
      '-map','0:v','-map','[a]','-t','8.333333','-c:v','libx264','-preset','fast','-crf','15','-pix_fmt','yuv420p','-c:a','aac','-ar','48000','-b:a','256k','-movflags','+faststart',str(out)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 native=np.diag([w/2688,h/1536,1]);scale=np.diag([2.,2.,1])
 for fi in range(200):
    idx=min(len(transforms)-1,round(max(0,float(map_to_source(fi/24)))*24))
    while source_idx<idx:
        ok,cur=cap.read()
        if not ok:raise ValueError('Truncated input')
        source_idx+=1
    trans=scale @ transforms[idx] @ np.linalg.inv(scale) @ native
    plate=cv.warpPerspective(cur,trans,(w,h),flags=cv.INTER_CUBIC)
    valid=(cv.cvtColor(plate,cv.COLOR_BGR2GRAY)>3).astype(np.uint8)
    # Feather only inside the source coverage so no dark boundary can enter the composite.
    distance=cv.distanceTransform(valid,cv.DIST_L2,3)
    alpha=np.minimum(1,distance/24.)*mask_near.astype(np.float32)/255
    if idx>141:
        # The lens pushes past the spoon. Reconstruct the clear air after the bead has departed.
        phase=np.clip((idx-142)/9,0,1)
        vertical=np.clip((yn-173)/29,0,1)
        alpha*=1-phase*(1-vertical)
    a=alpha[:,:,None];frame=(plate*a+clean*(1-a)).astype(np.uint8)
    proc.stdin.write(frame.tobytes())
 proc.stdin.close()
 if proc.wait():raise RuntimeError('Encoding failed')
 cap.release()
 print(out,flush=True)
if __name__=='__main__':assemble()
