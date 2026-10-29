"""Render reviewed identity overlays and encode NVENC, with CUDA pixel math.

The bitmap font is constant UI data. No CPU decoder, image transform or video
encoder is used; the host only transfers raw NV12 bytes to the encoder pipe.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from pipeline.cuda.core import CudaOnly, Video, tensor
from pipeline.cuda.recovery import saved_tensors
from pipeline.numeric import cuda_numbers
from training.common.files import write_json
from training.common.provenance import file_hash

FONT = {
    "A": "01110/10001/10001/11111/10001/10001/10001",
    "B": "11110/10001/10001/11110/10001/10001/11110",
    "C": "01111/10000/10000/10000/10000/10000/01111",
    "D": "11110/10001/10001/10001/10001/10001/11110",
    "E": "11111/10000/10000/11110/10000/10000/11111",
    "F": "11111/10000/10000/11110/10000/10000/10000",
    "G": "01111/10000/10000/10111/10001/10001/01111",
    "H": "10001/10001/10001/11111/10001/10001/10001",
    "I": "11111/00100/00100/00100/00100/00100/11111",
    "J": "00111/00010/00010/00010/10010/10010/01100",
    "K": "10001/10010/10100/11000/10100/10010/10001",
    "L": "10000/10000/10000/10000/10000/10000/11111",
    "M": "10001/11011/10101/10101/10001/10001/10001",
    "N": "10001/11001/10101/10011/10001/10001/10001",
    "O": "01110/10001/10001/10001/10001/10001/01110",
    "P": "11110/10001/10001/11110/10000/10000/10000",
    "Q": "01110/10001/10001/10001/10101/10010/01101",
    "R": "11110/10001/10001/11110/10100/10010/10001",
    "S": "01111/10000/10000/01110/00001/00001/11110",
    "T": "11111/00100/00100/00100/00100/00100/00100",
    "U": "10001/10001/10001/10001/10001/10001/01110",
    "V": "10001/10001/10001/10001/10001/01010/00100",
    "W": "10001/10001/10001/10101/10101/10101/01010",
    "X": "10001/10001/01010/00100/01010/10001/10001",
    "Y": "10001/10001/01010/00100/00100/00100/00100",
    "Z": "11111/00001/00010/00100/01000/10000/11111",
    "0": "01110/10001/10011/10101/11001/10001/01110",
    "1": "00100/01100/00100/00100/00100/00100/01110",
    "2": "01110/10001/00001/00010/00100/01000/11111",
    "3": "11110/00001/00001/01110/00001/00001/11110",
    "4": "00010/00110/01010/10010/11111/00010/00010",
    "5": "11111/10000/10000/11110/00001/00001/11110",
    "6": "01110/10000/10000/11110/10001/10001/01110",
    "7": "11111/00001/00010/00100/01000/01000/01000",
    "8": "01110/10001/10001/01110/10001/10001/01110",
    "9": "01110/10001/10001/01111/00001/00001/01110",
    "#": "01010/01010/11111/01010/11111/01010/01010",
    "-": "00000/00000/00000/11111/00000/00000/00000",
    ".": "00000/00000/00000/00000/00000/00110/00110",
    " ": "00000/00000/00000/00000/00000/00000/00000",
    "/": "00001/00010/00010/00100/01000/01000/10000",
    "'": "00100/00100/00000/00000/00000/00000/00000",
}


def glyphs(text, device, scale=3):
    pieces = []
    for ch in text:
        if ch not in FONT:
            raise ValueError(f"Unsupported label glyph {ch}")
        bitmap = tensor(
            [[int(bit) for bit in row] for row in FONT[ch].split("/")],
            device=device,
            dtype=torch.float32,
        )
        pieces.append(F.pad(bitmap, (0, 1)))
    word = torch.cat(pieces, 1)[None, None]
    return F.interpolate(word, scale_factor=scale, mode="nearest")[0, 0].bool()


def nv12(image):
    rgb = image.float()
    # Limited-range BT.709; pooling chroma and all RGB arithmetic stay CUDA.
    r, g, b = rgb.unbind(0)
    y = (16 + 0.182586 * r + 0.614231 * g + 0.062007 * b).round().clamp(16, 235).byte()
    u = 128 - 0.100644 * r - 0.338572 * g + 0.439216 * b
    v = 128 + 0.439216 * r - 0.398942 * g - 0.040274 * b
    uv = F.avg_pool2d(torch.stack((u, v))[None], 2, 2)[0].round().clamp(16, 240).byte()
    return torch.cat((y.flatten(), uv.permute(1, 2, 0).contiguous().flatten()))


def export_video(run, output, *, segmentation=True):
    run = Path(run).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((run / "run.json").read_text())
    if (
        manifest["status"] != "completed"
        or not manifest["identity_resolution"]["reviewed_initialization"]
    ):
        raise ValueError("A completed explicitly reviewed demo is required")
    device = "cuda:0"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback forbidden")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    source = manifest["source"]
    w, h = source["width"], source["height"]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "nv12",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(source["nominal_fps"]),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "16",
        "-b:v",
        "0",
        "-bf",
        "0",
        "-pix_fmt",
        "nv12",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    guard = CudaOnly()
    cache = {}
    frames = 0
    try:
        video = Video(source["path"], device)
        with (run / "observations.jsonl").open() as stream:
            for index, _, original in video.frames():
                row = json.loads(next(stream))
                assert row["frame_index"] == index
                with (
                    torch.inference_mode(),
                    torch.device(device),
                    cuda_numbers(device),
                    guard,
                ):
                    image = original.clone()
                    masks = saved_tensors(row, device)[3] if segmentation else None
                    rectangles = []
                    for i, p in enumerate(row["persons"]):
                        who = p.get("resolved_identity")
                        if not who:
                            continue
                        color = tensor(
                            [184, 126, 255]
                            if who["team_id"] == "sacramento_kings"
                            else [80, 224, 255],
                            device=device,
                            dtype=torch.float32,
                        )[:, None, None]
                        if masks is not None:
                            image = torch.where(
                                masks[i][None],
                                (image.float() * 0.82 + color * 0.18).round().byte(),
                                image,
                            )
                        a, b, c, d = (
                            tensor(p["bbox"], device=device).round().long().tolist()
                        )
                        image[:, b:d, a : a + 3] = color.byte()
                        image[:, b:d, c - 3 : c] = color.byte()
                        image[:, b : b + 3, a:c] = color.byte()
                        image[:, d - 3 : d, a:c] = color.byte()
                        names = who["player_name"].upper().split()
                        surname = (
                            " ".join(names[-2:])
                            if names[-1] in ("JR.", "JR")
                            else names[-1]
                        )
                        label = f"#{who['number']} {surname}"
                        if label not in cache:
                            cache[label] = glyphs(label, device)
                        mask = cache[label]
                        th, tw = mask.shape
                        left = (
                            ((tensor(a, device=device) + c - tw) / 2)
                            .round()
                            .clamp(3, w - tw - 9)
                        )
                        top = (tensor(b, device=device) - th - 12).clamp(2, h - th - 12)
                        for lane in range(5):
                            rect = torch.stack((left, top, left + tw + 8, top + th + 8))
                            if not rectangles:
                                break
                            previous = torch.stack(rectangles)
                            overlap = (
                                (rect[0] < previous[:, 2])
                                & (rect[2] > previous[:, 0])
                                & (rect[1] < previous[:, 3])
                                & (rect[3] > previous[:, 1])
                            )
                            if not bool(overlap.any()):
                                break
                            top = (top - th - 10).clamp_min(2)
                        rectangles.append(rect)
                        x, y, right, bottom = rect.long().tolist()
                        background = image[:, y:bottom, x:right]
                        background.copy_((background.float() * 0.18).round().byte())
                        image[:, bottom - 2 : bottom, x:right] = color.byte()
                        patch = image[:, y + 3 : y + 3 + th, x + 4 : x + 4 + tw]
                        patch.copy_(
                            torch.where(mask[None], torch.full_like(patch, 250), patch)
                        )
                    heading = "SAC - SAS / EXTRAIT ASSISTE VERIFIE"
                    if heading not in cache:
                        cache[heading] = glyphs(heading, device, 2)
                    mask = cache[heading]
                    th, tw = mask.shape
                    image[:, 12 : 12 + th + 12, w - tw - 28 : w - 8] = 20
                    patch = image[:, 18 : 18 + th, w - tw - 18 : w - 18]
                    patch.copy_(
                        torch.where(mask[None], torch.full_like(patch, 235), patch)
                    )
                    payload = nv12(image).cpu()
                    preview = (
                        image.permute(1, 2, 0).contiguous().cpu()
                        if index in (0, 25, 50, 65, 75, 100, 103, 115, 124)
                        else None
                    )
                encoder.stdin.write(payload.numpy().tobytes())
                if preview is not None:
                    from PIL import Image

                    Image.fromarray(preview.numpy()).save(
                        output.with_name(f"{output.stem}-{index:03d}.png")
                    )
                frames += 1
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        code = encoder.wait()
        video.close()
        if code:
            raise RuntimeError(error)
        if frames != manifest["frames"]:
            raise ValueError("Rendered frame count mismatch")
        report = {
            "source_run": str(run),
            "source_manifest_sha256": file_hash(run / "run.json"),
            "output": str(output),
            "sha256": file_hash(output),
            "frames": frames,
            "encoder": "h264_nvenc",
            "decoder": "NVDEC",
            "pixel_math": "CUDA",
            "font": "constant_5x7_bitmap",
            "reviewed_initialization": True,
            "cuda_guard": {
                **dict(guard.calls),
                "cpu_tensor_calculations": 0,
                "fallback_events": [],
            },
            "command": command,
        }
        write_json(output.with_suffix(".json"), report)
        return report
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    print(json.dumps(export_video(a.run, a.output), indent=2))
