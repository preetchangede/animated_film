"""Split the original H3 graph at the joint AV sampler. Keep the finishing graph by take."""
import copy
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/content/drip')
OUTPUT = Path('/content/minimax-h3/h3/ComfyUI/output')
HTTP = 'http://127.0.0.1:8188'

def node(graph, kind):
    keys = [key for key, item in graph.items() if item['class_type'] == kind]
    if len(keys) != 1:
        raise ValueError(f'Expected one {kind} node, got {keys}')
    return keys[0]

def subset(graph, roots):
    keep = set()
    def visit(key):
        if key in keep:
            return
        keep.add(key)
        for ref in graph[key]['inputs'].values():
            if isinstance(ref, list) and len(ref) == 2 and str(ref[0]) in graph:
                visit(str(ref[0]))
    for key in roots:
        visit(key)
    return {key: graph[key] for key in graph if key in keep}

def prepare(graph, stem, folder):
    folder.mkdir(exist_ok=True, parents=True)
    result = copy.deepcopy(graph)
    stage = graph[node(graph, 'H3SeparateAVLatent')]['inputs']['latent']
    decode = node(graph, 'VAEDecode')
    video = node(graph, 'H3SaveVideoNVENC')
    assert graph[node(graph, 'VAEDecodeAudio')]['inputs']['samples'] == stage
    assert graph[stage[0]]['class_type'] == 'SamplerCustomAdvanced'
    proof = copy.deepcopy(graph)
    proof[decode]['inputs']['samples'] = stage
    proof[video]['inputs']['filename_prefix'] = 'drip-motion/' + stem
    proof['9010'] = {'class_type':'H3SaveAVLatent','inputs':{'latent':stage,'filename_prefix':'drip-motion/'+stem}}
    result['9011'] = {'class_type':'H3LoadAVLatent','inputs':{'path':'CHECKPOINT'}}
    for item in result.values():
        for key, ref in item['inputs'].items():
            if ref == stage:
                item['inputs'][key] = ['9011',0]
    result[video]['inputs']['filename_prefix'] = 'drip-finished/' + stem
    result = subset(result, [video])
    assert sum(v['class_type']=='SamplerCustomAdvanced' for v in result.values()) == 1
    tmp = folder / (stem+'.new')
    tmp.write_text(json.dumps(result))
    tmp.replace(folder / (stem+'.json'))
    return subset(proof, [video,'9010'])

def api(path, data=None):
    req = urllib.request.Request(HTTP+path, data=json.dumps(data).encode() if data else None,
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as f:
        return json.load(f)

def finish(stem):
    import shutil
    folder = ROOT/'stages'
    target = ROOT/'takes'/(stem+'.mp4')
    import fcntl
    lock = (folder/(stem+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if target.is_file():
        return
    graph = json.loads((folder/(stem+'.json')).read_text())
    files = sorted((OUTPUT/'drip-motion').glob(stem+'_*.safetensors'))
    if not files:
        raise FileNotFoundError(stem+' motion checkpoint')
    graph['9011']['inputs']['path'] = str(files[-1].relative_to(OUTPUT))
    # Use the delivery-size finish tested on shot 6. Keep the untouched high-resolution graph on disk.
    size = os.environ.get('DRIP_DETAIL_SIZE','2016')
    if size not in ('2016','2688'):
        raise ValueError('Supported detail widths are 2016 and 2688')
    if size == '2016':
        for kind in ('MiniMaxH3ImageToVideo','MiniMaxH3LatentUpscaleWithModel'):
            graph[node(graph,kind)]['inputs'].update(width=2016,height=1152)
    jobfile = folder/(stem+'.job')
    if jobfile.exists():
        job = jobfile.read_text().strip()
    else:
        job = api('/prompt',{'prompt':graph})['prompt_id']
        jobfile.write_text(job)
    log = ROOT/'takes'/(stem+'.jsonl')
    def report(message):
        with log.open('a') as f:
            print(json.dumps({'time':time.time(),'status':message}),file=f)
    start = time.monotonic()
    while True:
        rec = api('/history/'+job).get(job)
        if rec:
            if rec['status']['status_str'] != 'success':
                report('Error: detail pass failed for '+stem)
                raise RuntimeError(str(rec['status'])[:1800])
            break
        report(f'Refining reviewed motion · elapsed {time.monotonic()-start:.0f}s')
        time.sleep(12)
    outputs = rec['outputs'][node(graph,'H3SaveVideoNVENC')]
    mp4 = next(x for values in outputs.values() if isinstance(values,list) for x in values
               if isinstance(x,dict) and str(x.get('filename','')).endswith('.mp4'))
    file = OUTPUT/mp4.get('subfolder','')/mp4['filename']
    part = target.with_suffix('.partial.mp4')
    shutil.copyfile(file,part)
    os.replace(part,target)
    report(f'Full-quality finish complete · {time.monotonic()-start:.0f}s detail pass')
if __name__ == '__main__':
    import sys
    finish(sys.argv[1])
