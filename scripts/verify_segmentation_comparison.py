"""Re-score saved operational outputs and audit source/mask integrity."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.common.files import write_json
from training.common.provenance import ROOT, file_hash
from training.players.evaluation.metrics import match_frame, iou
from training.segmentation.masks import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.report/'protocol.json').read_text())
    records=[json.loads(l) for l in (ROOT/protocol['dataset']/'frames.jsonl').read_text().splitlines()]
    audit=json.loads((args.report/'data-audit.json').read_text())
    for name,sha in audit['all_source_annotation_hashes'].items():
        if file_hash(Path(name))!=sha:raise ValueError('Source annotations changed')
    recipes={}
    for name in ('baseline','box_prompt','direct_l','direct_x'):
        path=args.runs/name;result=json.loads((path/'result.json').read_text())
        for artifact,sha in result['artifacts_sha256'].items():
            if file_hash(path/artifact)!=sha:raise ValueError('Run artifact changed')
        predictions={r['frame_id']:r['detections'] for r in map(json.loads,(path/'predictions.jsonl').read_text().splitlines())}
        if set(predictions)!={r['frame_id'] for r in records}:raise ValueError('Frame coverage differs')
        counts=Counter();masks=Counter();fp_kind=Counter()
        for r in records:
            ds=predictions[r['frame_id']]
            players=[d for d in ds if d['role']=='player']
            gt=[b['source_bbox'] for b in r['boxes'] if b['class_id']==0]
            refs=[b['source_bbox'] for b in r['boxes'] if b['class_id']==1]
            events=match_frame(gt,players,score_threshold=.25,iou_threshold=.5)
            counts.update(e['kind'].lower() for e in events)
            for event in events:
                if event['kind']!='FP':continue
                d=players[event['prediction_index']];overlap=max((iou(d['bbox'],b) for b in gt),default=0)
                referee_overlap=max((iou(d['bbox'],b) for b in refs),default=0)
                kind='duplicate_player' if overlap>=.5 else 'overlaps_referee_gt' if referee_overlap>=.5 else 'partial_player_overlap' if overlap>=.1 else 'other_unmatched_instance'
                fp_kind[kind]+=1
            for d in ds:
                seg=d.get('segmentation')
                if seg:
                    validate(seg['mask'],r['width'],r['height'])
                    foreground=sum(seg['mask']['counts'][1::2])
                    if (seg['status']=='ok')!=(foreground>0):raise ValueError('Empty status mismatch')
                    masks[d['role']]+=1
                if name.startswith('direct') and d['role']=='referee' and seg:
                    raise ValueError('Direct path segmented a referee')
        expected=result['detection']['all']['player']
        if any(counts[k]!=expected[k] for k in ('tp','fp','fn')):raise ValueError('Re-scored counts differ')
        recipes[name]={'frames':len(predictions),'rescored_player_counts':dict(counts),
                       'mask_observations':dict(masks),'false_positive_geometry':dict(fp_kind),
                       'result_sha256':file_hash(path/'result.json')}
    write_json(args.report/'verification.json',{'status':'passed','source_annotations_unchanged':True,
        'recipes':recipes,'protocol_id':protocol['protocol_id']})
    print(json.dumps(recipes),flush=True)


if __name__=='__main__':main()
