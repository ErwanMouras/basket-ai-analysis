"""Container discard flags define displayed frames; missing frames still fail."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pipeline.cuda.core import CudaOnly, Video, tensor
from pipeline.numeric import cuda_numbers


class Decoder:
    def __init__(self, declared, decoded):
        self.declared = declared
        self.remaining = decoded
        self.image = torch.zeros((3, 2, 2), device="cuda:0")

    def __len__(self):
        return self.declared

    def seek_to_index(self, index):
        assert index == 0

    def get_batch_frames(self, size):
        count = min(size, self.remaining)
        self.remaining -= count
        return [self.image] * count


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class VideoTests(unittest.TestCase):
    def create(self, *, discard=True, declared=4, decoded=3):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "video.mp4"
        path.write_bytes(b"fixture")
        packets = [{"pts": p, "flags": "___"} for p in (0, 1, 3)]
        if discard:
            packets.append({"pts": 5, "flags": "_D_"})
        metadata = {
            "streams": [
                {
                    "avg_frame_rate": "25/1",
                    "time_base": "1/25",
                    "width": 2,
                    "height": 2,
                    "nb_frames": str(declared),
                }
            ],
            "packets": packets,
        }
        decoder = Decoder(declared, decoded)
        with (
            patch("PyNvVideoCodec.SimpleDecoder", return_value=decoder),
            patch(
                "subprocess.check_output", return_value=json.dumps(metadata).encode()
            ),
            torch.inference_mode(),
            cuda_numbers("cuda:0"),
            CudaOnly(),
        ):
            return Video(path)

    def test_discard_packet_does_not_create_a_phantom_final_frame(self):
        video = self.create()
        with torch.inference_mode(), cuda_numbers("cuda:0"), CudaOnly():
            rows = list(video.frames())
            self.assertTrue(torch.equal(video.timestamps, tensor([0, 1, 3]) / 25))
        self.assertEqual([row[0] for row in rows], [0, 1, 2])
        self.assertTrue(all(row[2].is_cuda for row in rows))
        self.assertEqual(video.source["declared_frames"], 4)
        self.assertEqual(video.source["presentation_frames"], 3)
        self.assertEqual(video.source["discarded_packets"], 1)

    def test_real_missing_presentation_frame_still_raises(self):
        video = self.create(decoded=2)
        with self.assertRaisesRegex(RuntimeError, "ended at frame 2, expected 3"):
            list(video.frames())

    def test_count_is_unchanged_without_discard_packets(self):
        video = self.create(discard=False, declared=3)
        self.assertEqual(len(list(video.frames())), 3)
        self.assertEqual(video.source["discarded_packets"], 0)

    def test_unexplained_metadata_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Container PTS"):
            self.create(declared=5)
