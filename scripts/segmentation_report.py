"""Render factual comparison tables, curves and source-aligned visual review cases."""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import sys

# Allow direct invocation from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from training.common.files import write_json, write_jsonl
from training.common.provenance import ROOT, file_hash
from training.players.evaluation.metrics import match_frame, operating_metrics, AREA_RANGES
from training.segmentation.masks import crop_mask

NAMES = {'baseline':'Témoin E-BARD bbox','box_prompt':'Bbox → SAM 2.1 Tiny','direct_l':'YOLO26 L segmentation','direct_x':'YOLO26 X segmentation'}


def load_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def reduced(ds):
    return [{k:d[k] for k in ('bbox','confidence','role')} for d in ds]


def as_metrics(ds):
    return [{'bbox':d['bbox'],'confidence':d['confidence'],'class_id':1 if d['role']=='referee' else 0} for d in ds if d['role']!='unknown']


def write_tables(output, results):
    rows=[]
    for recipe,result in results.items():
        for cohort,roles in result['detection'].items():
            for role,m in roles.items():
                rows.append({'recipe':recipe,'cohort':cohort,'role':role,**m})
    with (output/'detection.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    report=args.report; visuals=args.runs/'review'; visuals.mkdir(exist_ok=True)
    protocol=json.loads((report/'protocol.json').read_text())
    root=ROOT/protocol['dataset']; records=load_rows(root/'frames.jsonl')
    results={name:json.loads((args.runs/name/'result.json').read_text()) for name in NAMES}
    if any(r['status']!='finished' or r['protocol_id']!=protocol['protocol_id'] for r in results.values()):
        raise ValueError('Incomplete or incomparable results')
    for recipe,result in results.items():
        for name,digest in result['artifacts_sha256'].items():
            if file_hash(args.runs/recipe/name)!=digest:
                raise ValueError('Raw evidence hash mismatch')
    predictions={name:{r['frame_id']:r for r in load_rows(args.runs/name/'predictions.jsonl')} for name in NAMES}
    ocr={name:load_rows(args.runs/name/'ocr.jsonl') for name in NAMES}
    inherited_equal=all(reduced(predictions['baseline'][r['frame_id']]['detections'])==reduced(predictions['box_prompt'][r['frame_id']]['detections']) for r in records)
    if not inherited_equal:
        raise ValueError('Prompted path unexpectedly altered detector boxes/roles')
    summary={'schema_version':1,'protocol_id':protocol['protocol_id'],'recipes':results,
        'box_prompt_preserves_baseline_boxes_all_frames':inherited_equal,
        'raw_runs':str(args.runs),'mask_accuracy':None,'jersey_accuracy':None}
    write_json(report/'results.json',summary)
    write_tables(report,results)
    # Operational error cohorts use source areas, not resized network boxes.
    cohorts={}
    curves={}
    for name in NAMES:
        ps={r['frame_id']:as_metrics(predictions[name][r['frame_id']]['detections']) for r in records}
        cohorts[name]={area:operating_metrics(records,ps,score_threshold=.25,iou_threshold=.5,
                           area_range=bounds,class_id=0)[0] for area,bounds in AREA_RANGES.items()}
        low={r['frame_id']:predictions[name][r['frame_id']]['ap_detections'] for r in records}
        curves[name]=[{**operating_metrics(records,low,score_threshold=t,iou_threshold=.5,class_id=0)[0], 'threshold':t}
                      for t in (.05,.1,.2,.25,.3,.4,.5,.6,.7,.8,.9)]
    write_json(report/'source-area-metrics.json',cohorts)
    write_json(report/'threshold-curves.json',curves)
    # OCR reviewed-label template uses GT object IDs, never new tracker IDs.
    ocr_ids={f for group in protocol['subsets'].values() for f in group['ocr']}
    labels=[{'frame_id':r['frame_id'],'object_id':b['object_id'],'image_sha256':r['image_sha256'],
             'match_id':r['match_id'],'frame_index':r['frame_index'],'bbox':b['source_bbox'],
             'review_status':'pending','expected_number':None,'person_mask':None,'number_mask':None}
            for r in records if r['frame_id'] in ocr_ids for b in r['boxes'] if b['class_id']==0]
    write_jsonl(args.runs/'review-template.jsonl',labels)
    # Keep GT support in crop statistics: unmatched background reads are explicit.
    ocr_matched={}
    for name in NAMES:
        groups={'matched_player':[],'unmatched_prediction':[]}
        for crop in ocr[name]:
            record=next(r for r in records if r['frame_id']==crop['frame_id'])
            ds=predictions[name][crop['frame_id']]['detections']
            indices=[i for i,d in enumerate(ds) if d['role']=='player']
            boxes=[b['source_bbox'] for b in record['boxes'] if b['class_id']==0]
            events=match_frame(boxes,[ds[i] for i in indices],score_threshold=.25,iou_threshold=.5)
            matched={indices[e['prediction_index']] for e in events if e['kind']=='TP'}
            groups['matched_player' if crop['prediction_index'] in matched else 'unmatched_prediction'].append(crop)
        from training.segmentation.evaluate import ocr_summary
        ocr_matched[name]={group:ocr_summary(rows) for group,rows in groups.items()}
    write_json(report/'ocr-matched-support.json',ocr_matched)
    # Four preselected first OCR frames, plus worst detection error frame per match.
    cases=[]
    for match in sorted({r['match_id'] for r in records}):
        rs=sorted([r for r in records if r['match_id']==match],key=lambda r:r['frame_index'])
        chosen=[('first',rs[0])]
        scored=[]
        for r in rs:
            boxes=[b['source_bbox'] for b in r['boxes'] if b['class_id']==0]
            score=0
            for name in ('direct_l','direct_x'):
                ds=[d for d in predictions[name][r['frame_id']]['detections'] if d['role']=='player']
                events=match_frame(boxes,ds,score_threshold=.25,iou_threshold=.5)
                score+=sum(e['kind'] in ('FP','FN') for e in events)
            scored.append((score,-r['frame_index'],r))
        worst=max(scored,key=lambda x:(x[0],x[1]))[2]
        if worst['frame_id']!=rs[0]['frame_id']:chosen.append(('worst',worst))
        for reason,r in chosen:
            image=cv2.imread(str(root/r['image'])); panels=[]
            for name in NAMES:
                canvas=image.copy();ds=predictions[name][r['frame_id']]['detections']
                for d in ds:
                    seg=d.get('segmentation')
                    if seg and seg['status']=='ok':
                        a,b,c,dv=seg['mask']['roi'];mask=crop_mask(seg['mask'],[a,b,c,dv])
                        patch=canvas[b:dv,a:c]; color=np.array([180,80,220] if d['role']=='referee' else [230,130,30])
                        patch[mask]=(patch[mask]*.55+color*.45).astype(np.uint8)
                gt=[b['source_bbox'] for b in r['boxes'] if b['class_id']==0]
                players=[d for d in ds if d['role']=='player']
                events=match_frame(gt,players,score_threshold=.25,iou_threshold=.5)
                matched={e['prediction_index'] for e in events if e['kind']=='TP'}
                for i,d in enumerate(players):
                    a,b,c,dv=map(round,d['bbox']);color=(255,170,30) if i in matched else (40,40,255)
                    cv2.rectangle(canvas,(a,b),(c,dv),color,3)
                for box in gt:
                    a,b,c,dv=map(round,box);cv2.rectangle(canvas,(a,b),(c,dv),(40,240,40),2)
                for d in ds:
                    if d['role']=='referee':
                        a,b,c,dv=map(round,d['bbox']);cv2.rectangle(canvas,(a,b),(c,dv),(220,50,220),3)
                panel=cv2.resize(canvas,(960,540));cv2.rectangle(panel,(0,0),(960,32),(10,10,10),-1)
                cv2.putText(panel,name,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                panels.append(panel)
            grid=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
            filename=f'{match}-{reason}-f{r["frame_index"]}.jpg'
            cv2.imwrite(str(visuals/filename),grid,[cv2.IMWRITE_JPEG_QUALITY,88])
            cases.append({'file':filename,'frame_id':r['frame_id'],'frame_index':r['frame_index'],
                          'match_id':match,'selection':reason,'image_sha256':r['image_sha256']})
    write_json(report/'review-cases.json',cases)
    # Standalone charts for inspection and export.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for i,(name,r) in enumerate(results.items()):
        p=r['performance'];m=r['detection']['all']['player']
        axes[0].scatter(p['median_ms'],m['ap50_95'],s=85,label=NAMES[name])
        axes[0].annotate(name,(p['median_ms'],m['ap50_95']),xytext=(5,5+i*2),textcoords='offset points',fontsize=8)
        axes[1].plot([x['recall'] for x in curves[name]],[x['precision'] for x in curves[name]],'o-',label=NAMES[name],markersize=3)
    axes[0].set(xlabel='Perception median (ms/frame)',ylabel='Player bbox AP@50:95',title='626 verified frames / RTX 5080',xlim=(0,62),ylim=(.70,.81))
    axes[1].set(xlabel='Player recall',ylabel='Player precision',title='Score thresholds .05–.90 / IoU .50')
    for ax in axes:ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(report/'quality-latency.png',dpi=160);plt.close(fig)
    page=['<!doctype html><meta charset="utf-8"><title>Segmentation review</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#151820;color:#eee}img{max-width:100%}a{color:#8bd}pre{white-space:pre-wrap}</style>',
          '<h1>Comparaison segmentation</h1><p>Vert : annotations joueur. Bleu : prédiction appariée. Rouge : prédiction sans appariement à IoU .50. Violet : arbitre. Les masques ne sont pas une vérité terrain.</p>',
          '<p>Les cas « first » sont présélectionnés. Les cas « worst » maximisent FP + FN des modèles directs : ils illustrent les échecs, pas la fréquence moyenne.</p>']
    for case in cases:
        page += [f'<h2>{html.escape(case["match_id"])} / frame {case["frame_index"]} / {case["selection"]}</h2>',f'<img src="{case["file"]}" loading="lazy">']
    ocr_cases=[]
    indexed={r['frame_id']:r for r in records}
    candidates=[]
    for name in NAMES:
        for crop in ocr[name]:
            raw,neutral=(crop['crops'][p] for p in ('disabled','background_neutral'))
            accepted=raw.get('accepted',False) or neutral.get('accepted',False)
            change=raw.get('reading',{}).get('text')!=neutral.get('reading',{}).get('text')
            if accepted or change:
                candidates.append((int(accepted)*2+int(change),name,crop))
    candidates=sorted(candidates,key=lambda x:(-x[0],x[1],x[2]['frame_id']))[:16]
    page.append('<h1>Exemples OCR sélectionnés</h1><p>Lectures acceptées ou modifiées : sélection qualitative, sans label de numéro validé. Une confiance élevée ne démontre pas la justesse.</p>')
    for i,(_,name,crop) in enumerate(candidates):
        r=indexed[crop['frame_id']];image=cv2.imread(str(root/r['image']))
        detection=predictions[name][crop['frame_id']]['detections'][crop['prediction_index']]
        canvas=np.full((380,800,3),30,np.uint8)
        a,b,c,d=map(round,crop['bbox']);person=image[b:d,a:c]
        if person.size:
            person=cv2.resize(person,(240,320));canvas[50:370,:240]=person
        for j,policy in enumerate(('disabled','background_neutral')):
            item=crop['crops'][policy]
            if not item['eligible']:continue
            a,b,c,d=item['bbox'];pixels=image[b:d,a:c].copy()
            if item['mask_audit']['status']=='applied':
                foreground=crop_mask(detection['segmentation']['mask'],[a,b,c,d]);pixels[~foreground]=127
            pixels=cv2.resize(pixels,(512,128));top=65+j*155;canvas[top:top+128,270:782]=pixels
            reading=item['reading'];text=f"{policy}: {reading['text']} / {reading['confidence']:.3f} / accepted={item['accepted']}"
            cv2.putText(canvas,text,(270,top-10),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        cv2.putText(canvas,f"{name} / {r['match_id']} / f{r['frame_index']}",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        filename=f'ocr-{i:02d}.jpg';cv2.imwrite(str(visuals/filename),canvas)
        page.append(f'<img src="{filename}" loading="lazy">')
        ocr_cases.append({'file':filename,'recipe':name,'frame_id':r['frame_id'],'prediction_index':crop['prediction_index'],'selection':'accepted_or_changed_not_representative'})
    write_json(report/'ocr-review-cases.json',ocr_cases)
    (visuals/'review.html').write_text('\n'.join(page))
    print(json.dumps({'report':str(report),'cases':len(cases),'ocr_review_objects':len(labels),'baseline_boxes_preserved':inherited_equal,'private_visual_review':str(visuals/'review.html')}),flush=True)


if __name__=='__main__':main()
