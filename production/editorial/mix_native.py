"""Mix the film's recurring physical rhythm using its H3 audio, native performances and continuous coastal ambience."""
from pathlib import Path
import json,subprocess
from scipy.io.wavfile import read,write
from scipy.signal import butter,sosfilt
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
here=Path(__file__).resolve().parent
root=here.parent;rr=root/'render-review';sr=48000
def wav(file,start=0,seconds=None):
    cmd=[get_ffmpeg_exe(),'-v','error','-ss',str(start),'-i',str(file)]
    if seconds is not None:cmd+=['-t',str(seconds)]
    cmd+=['-vn','-ac','2','-ar',str(sr),'-f','f32le','-']
    return np.frombuffer(subprocess.run(cmd,capture_output=True,check=True).stdout,np.float32).reshape(-1,2).copy()
def filter_audio(x,cut,kind):
    return sosfilt(butter(3,cut,btype=kind,fs=sr,output='sos'),x,axis=0).astype(np.float32)
def edges(x,attack=.025,release=.11):
    x=x.copy();a=min(len(x),round(attack*sr));b=min(len(x),round(release*sr))
    if a:x[:a]*=np.sin(np.linspace(0,np.pi/2,a))[:,None]**2
    if b:x[-b:]*=np.cos(np.linspace(0,np.pi/2,b))[:,None]**2
    return x
def peak(x,level):
    m=np.max(np.abs(x));return x*(level/max(.00001,m))
def add(mix,when,sound,gain=1,pan=0):
    at=round(when*sr);length=min(len(sound),len(mix)-at)
    if length>0:
        mix[at:at+length]+=sound[:length]*gain*np.sqrt(np.array([1-pan,1+pan]))
def run():
    cuts=json.loads((root/'decisions-live.json').read_text())
    shots=json.loads((root/'shots.json').read_text())
    assert all(cuts.get(str(s['shot']),{}).get('status')=='Accepted' for s in shots)
    total=sum(s['seconds'] for s in shots);bed=np.zeros((total*sr,2),np.float32)
    ocean=wav(rr/'video_01_t01.mp4',.16,7.60)
    # Overlap different parts of a real surf take, rotating starting phase and stereo perspective.
    rng=np.random.default_rng(4);t=0;n=0
    while t<total:
        length=rng.uniform(3.4,5.7);i=round(rng.uniform(.05,7.45-length)*sr)
        part=ocean[i:i+round(length*sr)]
        if n%3==1:part=part[:,::-1]
        add(bed,t,edges(part,.55,.55),1)
        t+=length-.55;n+=1
    wet=filter_audio(bed,5100,'lowpass');inner=filter_audio(bed,1850,'lowpass')
    position=np.arange(len(bed))/sr
    perspective=np.interp(position,[0,6.6,7.4,61,63,68.2,70,85,88,94,96,119.9],
                           [1,1,.17,.17,.31,.31,.15,.12,.065,.065,.18,.2])[:,None]
    mix=(peak(inner,.055)*(1-perspective)+peak(wet,.13)*perspective).astype(np.float32)
    offsets={};pos=0
    for s in shots:
        n=s['shot'];o=cuts[str(n)];offsets[n]=pos;dur=s['seconds']
        src=wav(Path(o['file']),o.get('in',0),dur+.16)
        # Use original synced Foley on inserts. Reserve the pitch range for Granddad on his recurring phrase.
        if n in (2,6,7,13,15):
            foley=filter_audio(src,580,'lowpass')+.4*filter_audio(src,4300,'highpass')
            layer=peak(foley,{2:.10,6:.065,7:.06,13:.060,15:.081}[n])
        elif n in (1,9):
            layer=peak(filter_audio(src,2600,'lowpass'),.035 if n==1 else .06)
        else:
            layer=peak(src,{3:.23,4:.23,5:.13,8:.10,10:.12,11:.14,12:.26,14:.22}[n])
        add(mix,pos,edges(layer,.08,.15))
        pos+=dur
    # The exact same real objects recur. Amplitude and perspective vary as the boy moves.
    first=peak(edges(wav(Path(cuts['3']['file']),.68,.57),.015,.15),.23)
    iron_time=float(cuts['4'].get('iron_onset',2.85))
    second=peak(edges(wav(Path(cuts['4']['file']),max(0,iron_time-.07),.57),.012,.15),.24)
    cup=peak(wav(here/'cup-strike.wav'),.20)
    phrase=peak(wav(here/'granddad-phrase.wav'),.19);answer=peak(wav(here/'granddad-answer.wav'),.20)
    for when,intensity in [(35.1,.52),(40.1,.74),(44.3,.43),(48.6,.62),(53.6,.45),(72,.40)]:
        add(mix,when,first,intensity,-.25);add(mix,when+.74,second,intensity*.95,-.05)
    add(mix,9.95,phrase,.73,.17)
    add(mix,41.12,edges(phrase[:round(sr*1.85)],.02,.11),.71,.2)
    add(mix,94.23,answer,.86,.14)
    ending=float(cuts['15'].get('final_beat',3.55))+offsets[15]
    add(mix,ending,first,.62,-.25);add(mix,ending+.72,second,.62,-.05);add(mix,ending+1.46,cup,.76,.22)
    add(mix,ending+1.75,edges(answer[:round(2.2*sr)],.07,.28),.67,.13)
    mixed=filter_audio(mix,35,'highpass');mixed=edges(mixed,.26,.72)
    write(here/'native-mix.wav',sr,(np.clip(mixed,-.95,.95)*32767).astype(np.int16))
if __name__=='__main__':run()
