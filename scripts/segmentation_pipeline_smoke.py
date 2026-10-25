"""Real-model source-video smoke, with mask persistence and replay checks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import load_config
from pipeline.run import analyze, recompute
from training.common.files import write_json
from training.segmentation.masks import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--frames',type=int,default=30)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--recipe',choices=('baseline','box-prompt','direct-l','direct-x'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    summary=[]
    for name in ([args.recipe] if args.recipe else ('box-prompt','direct-l','direct-x')):
        cfg=load_config(f'configs/segmentation/{name if name != "baseline" else "box-prompt"}.yaml',overrides={'max_frames':args.frames})
        if name=='baseline':
            cfg['segmentation']['strategy']='disabled'
            cfg['segmentation']['weights']=None
            cfg['jersey']['mask_policy']='disabled'
        for section in ('people','pose'):cfg[section]['device']=args.device
        path=args.output/name
        analyze('datas/train/HOU_LAL/G1/video.mp4',path,cfg,
                roster_path='training/jersey/examples/hou_lal_g1_2026/jersey.json')
        run=json.loads((path/'run.json').read_text())
        observations=[json.loads(l) for l in (path/'observations.jsonl').read_text().splitlines()]
        masks=refs=player_obs=ref_masks=0
        for frame in observations:
            for p in frame['persons']:
                seg=p.get('segmentation')
                if seg:validate(seg['mask'],frame['width'],frame['height']);masks+=1
                if p['role']=='referee':refs+=1;ref_masks+=int(seg is not None)
                if p['role']=='player':player_obs+=1
        assert run['status'] in ('completed','partial'),run['status']
        assert all(run['capabilities'][key]=='ok' for key in cfg['required']),run['capabilities']
        assert run['capabilities']['ball']=='disabled' and run['capabilities']['court']=='disabled'
        assert len(observations)==args.frames
        assert refs>0 and player_obs>0
        assert (masks==0) if name=='baseline' else (masks>0)
        if name.startswith('direct') or name=='baseline':assert ref_masks==0
        else:assert ref_masks==refs
        recompute(path,path/'recomputed.json')
        summary.append({'recipe':name,'status':run['status'],'frames':len(observations),
                        'masks':masks,'player_observations':player_obs,
                        'referee_observations':refs,'referee_masks':ref_masks,'recompute':'passed',
                        'timings':run.get('timings'),'resources':run.get('resources'),
                        'observation_bytes':(path/'observations.jsonl').stat().st_size,
                        'ocr_log_rows':sum(1 for _ in (path/'jersey_reads.jsonl').open()),
                        'confirmed_observations':sum(p.get('jersey',{}).get('status')=='confirmed' for f in observations for p in f['persons'])})
        print(json.dumps(summary[-1]),flush=True)
    write_json(args.output/'summary.json',{'status':'passed','device':args.device,'results':summary})


if __name__=='__main__':main()
