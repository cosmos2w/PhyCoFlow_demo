import numpy as np
import torch as th
import torch.nn as nn
from functools import partial
from tqdm import tqdm


class sde:
    """SDE solver class"""
    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat

    def __forward_fn(self):
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }
        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Sampler type not implemented.")
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)
        return samples


class ode:
    """ODE solver class (fixed-step Euler/Heun only, no torchdiffeq dependency)"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol=1e-6,
        rtol=1e-3,
    ):
        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device
        samples = [x]

        for ti in self.t[:-1]:
            t_vec = th.ones(x[0].size(0) if isinstance(x, tuple) else x.size(0)).to(device) * ti
            drift = self.drift(x, t_vec, model, **model_kwargs)

            if self.sampler_type == "Heun" and ti < self.t[-2]:
                x_euler = x + drift * self.dt if not isinstance(x, tuple) else tuple(
                    xi + di * self.dt for xi, di in zip(x, drift))
                t_next = th.ones_like(t_vec) * (ti + self.dt)
                drift2 = self.drift(x_euler, t_next, model, **model_kwargs)
                if isinstance(x, tuple):
                    x = tuple(xi + 0.5 * (d1 + d2) * self.dt for xi, d1, d2 in zip(x, drift, drift2))
                else:
                    x = x + 0.5 * (drift + drift2) * self.dt
            else:
                if isinstance(x, tuple):
                    x = tuple(xi + di * self.dt for xi, di in zip(x, drift))
                else:
                    x = x + drift * self.dt

            samples.append(x)

        return samples
