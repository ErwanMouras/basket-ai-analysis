"""The strict CUDA policy excludes even ONNX CPU graph partitions."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from training.players.pose import create_session, settings


class StrictPoseTests(unittest.TestCase):
    def test_cpu_execution_provider_is_disabled_explicitly(self):
        options = SimpleNamespace(add_session_config_entry=Mock())
        session = SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider'], disable_fallback=Mock())
        runtime = SimpleNamespace(__version__='1.27.0', SessionOptions=lambda: options,
                                  preload_dlls=Mock(), get_available_providers=lambda: ['CUDAExecutionProvider'],
                                  InferenceSession=Mock(return_value=session))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))

        def version(package):
            if package == 'onnxruntime-gpu':
                return '1.27.0'
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(package)

        with patch.dict(sys.modules, {'torch': torch, 'onnxruntime': runtime}), \
             patch('training.players.pose.importlib.metadata.version', side_effect=version):
            create_session('fixture.onnx', settings({'device': 'cuda:0', 'allow_cpu_fallback': False}))
        options.add_session_config_entry.assert_called_once_with('session.disable_cpu_ep_fallback', '1')
        providers = runtime.InferenceSession.call_args.kwargs['providers']
        self.assertEqual([provider[0] for provider in providers], ['CUDAExecutionProvider'])
        session.disable_fallback.assert_called_once_with()
