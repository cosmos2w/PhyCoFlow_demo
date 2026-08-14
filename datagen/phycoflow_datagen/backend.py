"""Small NumPy/PyTorch FFT backend used by all benchmark solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ArrayBackend:
    """Expose only the array operations needed by the spectral solvers.

    Random initial conditions are always created with NumPy before transfer so a
    seed identifies the same initial field on CPU and GPU.
    """

    name: str
    device: str
    dtype_name: str

    def __post_init__(self) -> None:
        if self.dtype_name not in {"float32", "float64"}:
            raise ValueError("solver dtype must be float32 or float64")
        if self.name == "numpy":
            if self.device != "cpu":
                raise ValueError("the NumPy backend supports only --device cpu")
            self.module = np
            self.real_dtype = np.float32 if self.dtype_name == "float32" else np.float64
            self.complex_dtype = np.complex64 if self.dtype_name == "float32" else np.complex128
            self.device_description = "CPU (NumPy)"
            return
        if self.name != "torch":
            raise ValueError(f"unknown backend: {self.name}")

        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        self.module = torch
        self.real_dtype = torch.float32 if self.dtype_name == "float32" else torch.float64
        self.complex_dtype = torch.complex64 if self.dtype_name == "float32" else torch.complex128
        if self.device.startswith("cuda"):
            logical_index = torch.device(self.device).index or 0
            self.device_description = f"{self.device} ({torch.cuda.get_device_name(logical_index)})"
        else:
            self.device_description = f"{self.device} (PyTorch)"

    def asarray(self, value: Any, *, complex_: bool = False):
        dtype = self.complex_dtype if complex_ else self.real_dtype
        if self.name == "numpy":
            return np.asarray(value, dtype=dtype)
        return self.module.as_tensor(value, dtype=dtype, device=self.device)

    def arange(self, start: int, stop: int):
        if self.name == "numpy":
            return np.arange(start, stop, dtype=self.real_dtype)
        return self.module.arange(start, stop, dtype=self.real_dtype, device=self.device)

    def fftfreq(self, n: int, d: float):
        if self.name == "numpy":
            return np.fft.fftfreq(n, d=d).astype(self.real_dtype)
        return self.module.fft.fftfreq(n, d=d, dtype=self.real_dtype, device=self.device)

    def fft(self, value):
        return np.fft.fft(value) if self.name == "numpy" else self.module.fft.fft(value)

    def ifft(self, value):
        return np.fft.ifft(value) if self.name == "numpy" else self.module.fft.ifft(value)

    def fft2(self, value):
        return np.fft.fft2(value) if self.name == "numpy" else self.module.fft.fft2(value)

    def ifft2(self, value):
        return np.fft.ifft2(value) if self.name == "numpy" else self.module.fft.ifft2(value)

    def exp(self, value):
        return self.module.exp(value)

    def mean(self, value, axis: int):
        if self.name == "numpy":
            return np.mean(value, axis=axis)
        return self.module.mean(value, dim=axis)

    def meshgrid_xy(self, x, y):
        return self.module.meshgrid(x, y, indexing="xy")

    def zeros_like(self, value):
        return self.module.zeros_like(value)

    def to_numpy(self, value) -> np.ndarray:
        if self.name == "numpy":
            return np.asarray(value)
        return value.detach().cpu().numpy()

    def synchronize(self) -> None:
        if self.name == "torch" and self.device.startswith("cuda"):
            self.module.cuda.synchronize(self.device)

