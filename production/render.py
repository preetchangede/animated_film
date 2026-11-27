"""Render image comparisons or video takes. Invoke on a prepared film G4."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

root = Path('/content/drip')
sys.path.insert(0, '/content/minimax-h3')
from h3_gpu import detect_profile, verify_a100_driver
profile = detect_profile(colab=True)
os.environ['H3_GPU_PROFILE'] = profile
if profile == 'a100':
    verify_a100_driver()
    os.environ['SERVER_ATTENTION_BACKEND'] = 'sla'
    os.environ['SERVER_DENSE_ATTENTION_BACKEND'] = 'pytorch'
os.environ.setdefault('SERVER_MEMORY_PROFILE', 'dynamic')
from h3_ui import application as app
from h3_app.generation.h3 import generate
from h3_app.generation.h3_image import H3ImageRequest, generate_h3_image
from h3_app.generation.requests import H3Request

def run(stage, number, take, shots):
    shot = shots[number - 1]
    stem = f'{stage}_{number:02d}_t{take:02d}'
    suffix = '.png' if stage == 'image' else '.mp4'
    target = root / 'takes' / (stem+suffix)
    preview = stage == 'video' and number in range(1,16)
    if preview:
        if target.is_file():
            return
        target = root / 'stages' / (stem+'.mp4')
        target.parent.mkdir(exist_ok=True)
    if target.exists() and target.stat().st_size:
        print('Already rendered', target, flush=True)
        return
    seed = 74300 + number * 100 + take
    services, runtime = app._generation_services(), app._runtime_config()
    source = root / 'frames' / f'{number:02d}.png'
    if stage == 'image':
        refs = [str(source)] + [str(root / 'refs' / n) for n in shot['references']]
        req = H3ImageRequest(shot['image_prompt'],2048,1152,seed,tuple(refs),False)
        stream = generate_h3_image(req, services, runtime)
    else:
        values = app.storyboard_request_defaults()
        values.update(mode='First / last frame', first_image=str(source), last_image=None,
                      prompt=shot['video_prompt'], duration=shot['seconds']+(0.20 if shot['seconds']==7 else 0.375), seed=seed,
                      model_profile='Original', generation_mode='Normal', steps=18,
                      scheduler='beta', attention_mode='PyTorch' if profile == 'a100' else 'Sage 2', cache_mode='Off',
                      width=2688, height=1536, auto_megapixels='4 MP',
                      text_encoder='INT8 ConvRot', lightvae_decoder='none',
                      use_sol_vae=False, use_lynnreal_vae=False, use_trt_vae=False,
                      stage_model_offload=profile == 'a100', result_format='Video', use_int8_vae=False,
                      latent_upscale=True, latent_upscaler_model='Quality (FP32)',
                      latent_upscale_refine_steps=2, latent_upscale_method='Full-frame refinement',
                      postprocess='None')
        values['prompt'] = values['prompt'].replace(
            'warm amber lamp at right, diffuse blue sea-window light.',
            'the same gentle lamplight and overcast blue window light stay steady for the entire shot. Fixed exposure and white balance.')
        if number == 1 and take >= 2:
            values['last_image'] = str(root / 'frames' / '01-end.png')
            values['prompt'] = (
                'The first frame <Picture 1> and last frame <Picture 2> show the same overcast coastal afternoon. '
                'Locked, sheltered viewpoint behind cool wet cottage glass. A little drop slides slowly down the left third of the pane. '
                'The indigo sea undulates in broad overlapping folded paper planes. The little white lighthouse stays in the far distance. '
                'The lighthouse lamp is unlit. The cloud cover stays dense and blue-grey. Constant faint overcast daylight throughout. '
                'Quiet, soft rain on glass and surf from beyond the window. No music, no voices.')
        if number == 14:
            landing = root / 'takes' / 'image_14_t01.png'
            if not landing.is_file():
                raise FileNotFoundError('Shot 14 needs its selected smile landing frame')
            values['last_image'] = str(landing)
            values['prompt'] = values['prompt'].replace(
                'is fully referenced.',
                'is fully referenced. At the final instant of the target video <Picture 2> is fully referenced as the ending pose. '
                'Hold the pursed whistle at the beginning; the smile in Picture 2 arrives only after the offscreen cup answers.',
                1)
        if number == 14 and take >= 2:
            values['prompt'] = (
                'At the start of the shot <Picture 1> is fully referenced. At the end <Picture 2> is fully referenced. '
                'A single uninterrupted, locked-off close-up of the seated older man. The amber light on his face and the soft blue light from the window '
                'are constant from beginning to end. His eyes stay shut. For six seconds his lips gently purse as he whistles a quiet phrase. '
                'He listens to one offscreen ceramic tap, breathes out through his nose, and the left corner of his closed mouth lifts, barely. '
                'Hold his knowing smile. Fixed exposure, matte paper, still background, practical kitchen lamp. '
                'Sound: quiet human whistle, rain outside, one delicate distant cup tap. No dialogue or music.')
        if preview:
            import importlib
            from staged import prepare
            module = importlib.import_module('h3_app.generation.h3')
            original = module.construct_graph
            module.construct_graph = lambda *a,**kw: prepare(original(*a,**kw),stem,target.parent)
        stream = generate(H3Request.from_values(values), services, runtime, run_context={})
    log = root / 'takes' / (stem+'.jsonl')
    with log.open('a') as out:
        for result, status in stream:
            print(status, flush=True)
            out.write(json.dumps({'time': time.time(), 'seed':seed,'status':status})+'\n')
            out.flush()
            if status.lower().startswith('error:'):
                raise RuntimeError(status)
            if result:
                output = Path(result[0] if isinstance(result,list) else result)
                if output.is_file() and output.suffix.lower()==suffix:
                    tmp=target.with_suffix('.partial'+suffix)
                    shutil.copyfile(output,tmp)
                    os.replace(tmp,target)
    if preview:
        module.construct_graph = original
        if target.is_file():
            with log.open('a') as out:
                out.write(json.dumps({'time':time.time(),'seed':seed,'status':'Motion proof ready for review'})+'\n')
    if not target.is_file():
        raise RuntimeError(f'No take produced for {stem}')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['image','video'])
    parser.add_argument('shots',nargs='+',type=int,choices=range(1,16))
    parser.add_argument('--take',type=int,default=1)
    args=parser.parse_args()
    (root/'takes').mkdir(exist_ok=True,parents=True)
    shots=json.loads((root/'shots.json').read_text())
    for number in args.shots:
        run(args.stage,number,args.take,shots)
