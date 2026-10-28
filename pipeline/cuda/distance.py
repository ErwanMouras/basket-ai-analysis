"""Conservative metric semantics with one fused CUDA integration per track."""

import torch

from pipeline.cuda.core import tensor
from pipeline.distance import Distance
from pipeline.numeric import cuda_numbers


class CudaDistance(Distance):
    def __init__(self, config, run_id):
        if not config.get("device", "cpu").startswith("cuda:"):
            raise RuntimeError("Native statistics require CUDA")
        super().__init__(config, run_id, defer_exports=True)
        self.zero_point = torch.zeros(2, device=self.device, dtype=torch.float64)
        self.params = tensor(
            [
                config[key]
                for key in (
                    "max_gap_s",
                    "max_speed_m_s",
                    "smoothing_tau_s",
                    "deadband_m",
                )
            ],
            device=self.device,
        )

    def update(self, frame):
        with cuda_numbers(self.device):
            return super().update(frame)

    def result(self):
        with cuda_numbers(self.device):
            return super().result()

    def _step(self, prev, state, row, person, dt):
        from pipeline.cuda.kernels import metric_step

        if prev is None:
            return None, None
        same = prev["subject"] == state["subject"]
        conflict = bool(state["conflict"]) or bool(prev["conflict"])
        missing = state["point"] is None or prev["point"] is None
        changed = prev["method"] != state["method"]
        flags = (
            int(same) | (int(conflict) << 1) | (int(missing) << 2) | (int(changed) << 3)
        )

        def point(value):
            return value if value is not None else self.zero_point

        result = metric_step(
            point(state["point"]),
            point(prev["point"]),
            point(prev["filtered"]),
            point(prev["anchor"]),
            dt,
            self.params,
            row,
            flags,
        )
        reason = int(result[5])  # export only the final control category
        if state["point"] is not None:
            state["filtered"], state["anchor"] = result[:2], result[2:4]
        labels = (
            None,
            "identity_transition_or_conflict",
            "timestamp_gap",
            person.get("position_status", "position_unavailable"),
            "ground_point_method_change",
            "implausible_speed",
        )
        return labels[reason], result[4] if reason == 0 else None
