import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import torch

from pipeline.config import load_config
from training.jersey.config import settings as jersey_settings
from training.jersey.crops import candidate
from training.segmentation.config import settings
from training.segmentation.adapters import SegmentedPeople
from training.segmentation.masks import encode, decode, crop_mask, validate
from training.segmentation.review import score_reviewed


def row(bbox=None, role="player"):
    return {"bbox": bbox or [0., 0., 200., 300.], "confidence": .95, "class_id": 0, "role": role}


def result(masks, boxes, prompts=None):
    class Boxes(SimpleNamespace):
        def __len__(self):
            return len(self.conf)
    return SimpleNamespace(masks=SimpleNamespace(data=torch.as_tensor(masks)), boxes=Boxes(
        xyxy=torch.tensor(boxes, dtype=torch.float32), conf=torch.tensor([.95]*len(boxes)),
        cls=torch.tensor(prompts if prompts is not None else [0]*len(boxes))))


class MaskTests(unittest.TestCase):
    def test_lossless_random_and_uniform_masks(self):
        rng = np.random.default_rng(0)
        for mask in (np.zeros((10, 20), bool), np.ones((10, 20), bool), rng.random((10,20))>.5):
            payload=encode(mask, [2.2,1.1,18.2,9.1])
            validate(payload,20,10)
            np.testing.assert_array_equal(decode(payload),mask[1:10,2:19])

    def test_crop_in_source_coordinates(self):
        m=np.ones((10,20),bool); p=encode(m,[4,3,10,8])
        c=crop_mask(p,[2,1,12,9])
        self.assertEqual(c.sum(),30)
        self.assertTrue(c[2:7,2:8].all())
        self.assertFalse(c[:2].any())

    def test_reject_corrupt_or_out_of_bounds_masks(self):
        p=encode(np.ones((10,20),bool),[0,0,20,10])
        for mutation in ({"counts":[0,201]},{"counts":[-1,201]},{"roi":[-1,0,20,10]},{"encoding":"rle"},{"counts":[False,200]}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate({**p,**mutation},20,10)

    def test_empty_box_rejected(self):
        with self.assertRaises(ValueError): encode(np.ones((10,20),bool),[30,0,40,10])


class ConfigurationTests(unittest.TestCase):
    def test_disabled_by_default(self):
        c=load_config()
        self.assertEqual(c['segmentation']['strategy'],'disabled')
        self.assertEqual(c['jersey']['mask_policy'],'disabled')

    def test_invalid_options(self):
        for changes in ({'strategy':'automatic'},{'strategy':'direct'},{'referee_iou':float('nan')},
                        {'resolution':True},{'segment_referees':1},{'class_roles':{'x':'player'}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): settings(changes)

    def test_recipes_are_opt_in(self):
        for name in ('box-prompt','direct-l','direct-x'):
            c=load_config('configs/segmentation/'+name+'.yaml')
            self.assertNotEqual(c['segmentation']['strategy'],'disabled')

    def test_invalid_jersey_policy(self):
        with self.assertRaises(ValueError): jersey_settings({'mask_policy':'silently_drop_digits'})
        with self.assertRaises(ValueError): jersey_settings({'min_mask_coverage':1.5})


class AdapterTests(unittest.TestCase):
    def make(self, strategy, ds, prediction, **overrides):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        p=Path(temporary.name)/'model.pt'; p.write_bytes(b'fixture')
        detector=Mock(roles={2:'player',3:'referee'}, provenance={'test':True})
        detector.predict.return_value=ds
        model=Mock(names={0:'person'}); model.predict.return_value=[prediction]
        cfg=settings({'strategy':strategy,'weights':str(p),'class_roles':{'0':'player'},
                      'person_role_policy':'non_referee_person_proxy',**overrides})
        people={'device':'cpu','resolution':640,'confidence':.25,'max_detections':100}
        adapter=SegmentedPeople(people,cfg,detector_factory=lambda _:detector,model_factory=lambda _:model)
        return adapter,detector,model

    def test_prompt_mapping_preserves_boxes_and_roles(self):
        masks=np.zeros((2,30,20),bool); masks[0,:,:10]=1; masks[1,:,10:]=1
        ds=[row([0,0,20,30]),row([0,0,20,30],'referee')]
        a,_,_=self.make('box_prompt',ds,result(masks,[[0,0,20,30]]*2,[1,0]))
        got=a.predict(np.zeros((30,20,3),np.uint8))
        self.assertEqual(got[0]['bbox'],[0,0,20,30]); self.assertEqual(got[1]['role'],'referee')
        self.assertTrue(decode(got[0]['segmentation']['mask'])[:,10:].all())
        self.assertTrue(decode(got[1]['segmentation']['mask'])[:,:10].all())

    def test_empty_detection_never_calls_sam(self):
        a,_,model=self.make('box_prompt',[],None)
        self.assertEqual(a.predict(np.zeros((30,20,3),np.uint8)),[])
        model.predict.assert_not_called()

    def test_sam_alignment_fail_closed(self):
        a,_,_=self.make('box_prompt',[row([0,0,20,30])],result(np.ones((2,30,20),bool),[[0,0,20,30]]*2))
        with self.assertRaises(RuntimeError): a.predict(np.zeros((30,20,3),np.uint8))

    def test_direct_requests_referees_only_and_suppresses_duplicates(self):
        referee=row([0,0,10,30],'referee')
        a,detector,_=self.make('direct',[referee],result(np.ones((2,30,20),bool),[[0,0,10,30],[10,0,20,30]]))
        got=a.predict(np.zeros((30,20,3),np.uint8))
        self.assertEqual(detector.predict.call_args.kwargs['roles'],{'referee'})
        self.assertEqual([d['role'] for d in got],['player','referee'])
        self.assertNotIn('segmentation',got[-1])
        self.assertEqual(got[0]['bbox'],[10,0,20,30])

    def test_person_cannot_silently_become_player(self):
        with self.assertRaises(ValueError):
            self.make('direct',[],None,person_role_policy='unknown')

    def test_low_confidence_referee_does_not_delete_person(self):
        referee=row([0,0,20,30],'referee'); referee['confidence']=.1
        a,_,_=self.make('direct',[referee],result(np.ones((1,30,20),bool),[[0,0,20,30]]))
        self.assertEqual(len(a.predict(np.zeros((30,20,3),np.uint8))),2)

    def test_box_only_pass_skips_sam(self):
        a,_,model=self.make('box_prompt',[row([0,0,20,30])],None)
        self.assertEqual(len(a.predict(np.zeros((30,20,3),np.uint8),masks=False)),1)
        model.predict.assert_not_called()


class CropTests(unittest.TestCase):
    def image(self):
        return np.random.default_rng(1).integers(0,256,(300,200,3),dtype=np.uint8)

    def test_mask_neutralizes_background_without_changing_framing_or_quality(self):
        image=self.image(); d=row(); m=np.ones((300,200),bool); m[:,100:110]=False
        d['segmentation']={'status':'ok','mask':encode(m,d['bbox'])}
        cfg=jersey_settings()
        raw=candidate(image,d,[d],cfg)
        masked=candidate(image,d,[d],{**cfg,'mask_policy':'require_mask'})
        self.assertEqual(raw['bbox'],masked['bbox'])
        self.assertEqual(raw['quality'],masked['quality'])
        np.testing.assert_array_equal(raw['color_image'],masked['color_image'])
        self.assertFalse(np.array_equal(raw['image'],masked['image']))
        self.assertEqual(masked['mask_audit']['status'],'applied')

    def test_missing_mask_fallback_and_strict_abstention(self):
        image=self.image(); d=row(); cfg=jersey_settings()
        c=candidate(image,d,[d],{**cfg,'mask_policy':'background_neutral'})
        self.assertEqual(c['mask_audit']['status'],'raw_fallback')
        self.assertIsNone(candidate(image,d,[d],{**cfg,'mask_policy':'require_mask'}))

    def test_small_foreground_does_not_erase_number_silently(self):
        image=self.image(); d=row(); m=np.zeros((300,200),bool); m[100,100]=True
        d['segmentation']={'status':'ok','mask':encode(m,d['bbox'])}
        cfg={**jersey_settings(),'mask_policy':'require_mask'}
        self.assertIsNone(candidate(image,d,[d],cfg))


class ReviewTests(unittest.TestCase):
    def fixture(self):
        r={'frame_id':'f','width':20,'height':30,'image_sha256':'hash','boxes':[{'object_id':'o','class_id':0,'source_bbox':[0,0,20,30]}]}
        d=row([0,0,20,30]); d['segmentation']={'mask':encode(np.ones((30,20),bool),d['bbox'])}
        c={'frame_id':'f','prediction_index':0,'crops':{p:{'accepted':True,'reading':{'text':'07'}} for p in ('disabled','background_neutral')}}
        label={'frame_id':'f','object_id':'o','image_sha256':'hash','review_status':'reviewed','expected_number':'07','person_mask':d['segmentation']['mask']}
        return [r],{'f':[d]},[c],[label]

    def test_gold_not_inferred_from_predictions(self):
        rs,ps,cs,labels=self.fixture(); labels[0]['review_status']='pending'
        got=score_reviewed(rs,ps,cs,labels)
        self.assertEqual(got['reviewed_objects'],0)
        self.assertIsNone(got['mask']['mean_iou'])
        self.assertIsNone(got['ocr']['disabled']['precision_when_read'])

    def test_exact_match_preserves_leading_zero(self):
        got=score_reviewed(*self.fixture())
        self.assertEqual(got['mask']['mean_iou'],1)
        self.assertEqual(got['ocr']['disabled']['correct'],1)

    def test_false_read_on_unreadable_is_wrong(self):
        rs,ps,cs,labels=self.fixture(); labels[0]['expected_number']=None
        got=score_reviewed(rs,ps,cs,labels)
        self.assertEqual(got['ocr']['disabled']['wrong'],1)

    def test_missed_detection_remains_in_denominator(self):
        rs,_,cs,labels=self.fixture(); got=score_reviewed(rs,{'f':[]},cs,labels)
        self.assertEqual(got['mask']['mean_iou'],0)
        self.assertEqual(got['ocr']['disabled']['correct_read_coverage'],0)

    def test_wrong_source_and_duplicate_gold_rejected(self):
        rs,ps,cs,labels=self.fixture()
        with self.assertRaises(ValueError): score_reviewed(rs,ps,cs,labels*2)
        labels[0]['image_sha256']='other'
        with self.assertRaises(ValueError): score_reviewed(rs,ps,cs,labels)


if __name__ == '__main__': unittest.main()
