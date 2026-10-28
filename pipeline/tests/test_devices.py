"""GPU failures fall back explicitly; corrupt models and inputs still fail."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from training.common.devices import DeviceFallback
from training.jersey.reader import ParseqReader, ResourceDeferred


class DeviceTests(unittest.TestCase):
    def test_gpu_inference_is_preferred(self):
        gpu = SimpleNamespace(provenance={"weights": "pinned"}, predict=Mock(return_value=[1]))
        factory = Mock(return_value=gpu)
        model = DeviceFallback(factory, "cuda:0")
        self.assertEqual(model.predict("image"), [1])
        factory.assert_called_once_with("cuda:0")
        self.assertEqual(model.provenance['device_policy']['active'], 'cuda:0')

    def test_gpu_inference_failure_retries_the_same_image_on_cpu_and_records_it(self):
        gpu = SimpleNamespace(provenance={}, predict=Mock(side_effect=RuntimeError('CUDA out of memory')))
        cpu = SimpleNamespace(provenance={"weights": "same"}, predict=Mock(return_value=[2]))
        model = DeviceFallback(lambda device: gpu if device.startswith('cuda') else cpu, 'cuda:0')
        provenance = model.provenance
        self.assertEqual(model.predict('frame'), [2])
        cpu.predict.assert_called_once_with('frame')
        self.assertIs(provenance, model.provenance)
        self.assertEqual(provenance['device_policy']['active'], 'cpu')
        self.assertEqual(len(provenance['device_policy']['events']), 1)

    def test_missing_gpu_at_initialization_uses_cpu(self):
        def factory(device):
            if device != 'cpu':
                raise ValueError('CUDA device unavailable')
            return SimpleNamespace(provenance={})
        self.assertEqual(DeviceFallback(factory, 'cuda:0').device, 'cpu')

    def test_onnx_device_exceptions_also_fall_back(self):
        class OnnxFailure(Exception):
            pass
        gpu = SimpleNamespace(provenance={}, predict=Mock(side_effect=OnnxFailure('CUDAExecutionProvider failure')))
        cpu = SimpleNamespace(provenance={}, predict=Mock(return_value=[3]))
        model = DeviceFallback(lambda device: gpu if device.startswith('cuda') else cpu, 'cuda:0')
        self.assertEqual(model.predict('frame'), [3])
        self.assertEqual(model.device, 'cpu')

    def test_unrelated_errors_and_disabled_fallback_are_not_hidden(self):
        for message, fallback in [('corrupt checkpoint', 'cpu'), ('CUDA error', None)]:
            factory = Mock(side_effect=RuntimeError(message))
            with self.assertRaises(RuntimeError):
                DeviceFallback(factory, 'cuda:0', fallback)
            factory.assert_called_once_with('cuda:0')

    def test_ocr_memory_pressure_uses_cpu_then_can_return_to_gpu(self):
        reader = ParseqReader.__new__(ParseqReader)
        reader.config = {'device': 'cuda:0', 'fallback_device': 'cpu'}
        reader.provenance = {'fallback_reads': {}}
        reader._cpu_reader = SimpleNamespace(read=Mock(return_value=['cpu']))
        reader._read_on_device = Mock(side_effect=[ResourceDeferred('gpu_memory_reserve'), ['gpu']])
        self.assertEqual(reader.read(['crop']), ['cpu'])
        self.assertEqual(reader.provenance['device'], 'cpu')
        self.assertEqual(reader.read(['crop']), ['gpu'])
        self.assertEqual(reader.provenance['device'], 'cuda:0')
        self.assertEqual(sum(reader.provenance['fallback_reads'].values()), 1)
