import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import torch
from pipeline.adapters import YoloBall
from pipeline.config import load_config


class Boxes:
    def __init__(self, scores, centers):
        self.conf=torch.tensor(scores)
        self.xywh=torch.tensor(centers)
    def __len__(self): return len(self.conf)


class YoloBallTests(unittest.TestCase):
    def adapter(self, boxes):
        adapter=YoloBall.__new__(YoloBall)
        adapter.config=load_config()['ball']
        adapter.model=Mock();adapter.model.predict.return_value=[SimpleNamespace(boxes=boxes)]
        return adapter

    def test_best_box_restores_source_coordinates_on_first_frame(self):
        adapter=self.adapter(Boxes([.3,.9],[[100,50,5,5],[256,144,5,5]]))
        frame=SimpleNamespace(index=0,image=np.zeros((1080,1920,3),np.uint8))
        result=adapter.predict([frame])
        self.assertEqual(result['position_px'],[960.,540.])
        self.assertEqual(result['source_frames'],[0])
        self.assertEqual(result['future_context_frames'],0)
        self.assertEqual(adapter.model.predict.call_args.kwargs['imgsz'],[288,512])
        self.assertEqual(adapter.model.predict.call_args.args[0].shape,(288,512,3))

    def test_no_detection_is_not_a_zero_position(self):
        adapter=self.adapter(Boxes([],[]))
        result=adapter.predict([SimpleNamespace(index=7,image=np.zeros((40,80,3),np.uint8))])
        self.assertIsNone(result['position_px'])
        self.assertIsNone(result['confidence'])
        self.assertEqual(result['status'],'not_detected')

    def test_only_current_frame_is_used(self):
        adapter=self.adapter(Boxes([],[]))
        frames=[SimpleNamespace(index=i,image=np.zeros((40,80,3),np.uint8)) for i in range(3)]
        self.assertEqual(adapter.predict(frames)['source_frames'],[2])

    def test_geometry_and_confidence_are_validated(self):
        for changes in ({'input_width':513},{'confidence':1.1},{'checkpoint_sha256':'wrong'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                load_config(overrides={'ball':{**load_config()['ball'],**changes}})
