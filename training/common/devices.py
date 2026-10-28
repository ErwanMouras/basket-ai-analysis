"""Explicit GPU preference with a recorded CPU fallback for device failures."""


def device_error(error):
    text = str(error).lower()
    return type(error).__name__ == "OutOfMemoryError" or any(
        token in text for token in ("cuda", "cudnn", "cublas", "gpu memory", "nvidia")
    )


class DeviceFallback:
    """Retry stateless perception on CPU, without hiding model/data errors."""

    def __init__(self, factory, preferred, fallback="cpu"):
        self.factory = factory
        self.preferred = self.device = preferred
        self.fallback = fallback
        self.policy = {"preferred": preferred, "active": preferred,
                       "fallback": fallback, "events": []}
        self.provenance = {}
        try:
            self.model = factory(preferred)
        except Exception as error:
            if not self._can_fallback(error):
                raise
            self._fallback(error)
        self._publish()

    def _can_fallback(self, error):
        return self.device.startswith("cuda:") and self.fallback == "cpu" and device_error(error)

    def _publish(self):
        self.provenance.clear()
        self.provenance.update(self.model.provenance)
        self.provenance["device_policy"] = self.policy

    def _fallback(self, error):
        previous = getattr(self, "model", None)
        if previous is not None and hasattr(previous, "close"):
            previous.close()
        self.model = None
        self.device = "cpu"
        self.policy["active"] = "cpu"
        self.policy["events"].append(f"{type(error).__name__}: {error}")
        self.model = self.factory("cpu")
        self._publish()

    def predict(self, *args, **kwargs):
        try:
            return self.model.predict(*args, **kwargs)
        except Exception as error:
            if not self._can_fallback(error):
                raise
            self._fallback(error)
            return self.model.predict(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "model"), name)
