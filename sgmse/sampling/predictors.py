import abc

import torch
import numpy as np

from sgmse.util.registry import Registry


PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(
            f"Debug update function not implemented for predictor {self}.")


@PredictorRegistry.register('onestep')
class OneStepPredictor(Predictor):
    def __init__(self, sde, score_fn, eps=0.03):
        super().__init__(sde, score_fn)
        self.eps = eps

    def update_fn(self, x, y, t, stepsize):
        def denoise_fn(score, x_t, sigma):
            # when t=eps=0.03,std=0.0188,sigma=t,sigma.pow(2)=0.00035344
            D = score * sigma.pow(2) + x_t  # equivalent to Eq. (10)
            return D
        f, g = self.rsde.discretize(x, y, t, stepsize)
        # f, g = sde.reverse(score_fn).discretize(x, y, t, stepsize)
        z = torch.randn_like(x)
        std = self.sde.marginal_prob(x, y, t)[1]  # std=sigma
        # grad = -self.score_fn(torch.cat([x, y], dim=1), t)
        grad = self.score_fn(x, y, t)
        # x_mean = x - f
        x_mean = denoise_fn(grad, x, std)
        x = x_mean + g[:, None, None, None] * z
        return x, x_mean


@PredictorRegistry.register('euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, y, t, *args):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        # probability_flow=True, while still use sde diffusion term
        f, g = self.rsde.ode(x, y, t, *args)
        # self.sde.reverse(score_fn).sde(x, y, t, *args)
        x_mean = x + f * dt
        x = x_mean + g[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean
        # step_size = (snr * std) ** 2 * 2  # this is theta
        # xt = x_mean + noise * \
        #     torch.sqrt(step_size * 2)[:, None, None, None]


@PredictorRegistry.register('heun_solver')
class HeunPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, y, t, *args):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)

        # First evaluation of the drift term
        # probability_flow=True, while still use sde diffusion term
        f1, g = self.rsde.ode(x, y, t, *args)

        # Predictor step using ODE (deterministic Euler step, no noise)
        x_pred = x + f1 * dt

        # Second evaluation of the drift term at predicted point
        # Note: time step for second evaluation
        t_next = t + dt
        f2, _ = self.rsde.ode(x_pred, y, t_next, *args)

        # Heun's corrector: average of the two drift evaluations
        f_avg = 0.5 * (f1 + f2)

        # Final update using averaged drift
        x_mean = x + f_avg * dt
        x = x_mean + g[:, None, None, None] * np.sqrt(-dt) * z

        return x, x_mean


@PredictorRegistry.register('mix_solver')
class MixPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, y, t, *args):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        # SDE
        f, g = self.rsde.sde(x, y, t, *args)
        # self.sde.reverse(score_fn).sde(x, y, t, *args)
        x_mean = x + f * dt
        sde_pred = x_mean + g[:, None, None, None] * np.sqrt(-dt) * z

        # ODE
        f1, g1 = self.rsde.ode(x, y, t, *args)
        # Predictor step using ODE (deterministic Euler step, no noise)
        ode_pred = x + f1 * dt

        return (ode_pred + sde_pred)/2, (ode_pred + sde_pred)/2


@PredictorRegistry.register('reverse_diffusion_with_dnn')
class ReverseDiffusionPredictor_dnn(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, y, t, stepsize):
        f, g = self.rsde.discretize(x, y, t, stepsize)
        # f, g = sde.reverse(score_fn).discretize(x, y, t, stepsize)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + g[:, None, None, None] * z
        return x, x_mean


@PredictorRegistry.register('reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, y, t, stepsize):
        f, g = self.rsde.discretize(x, y, t, stepsize)
        # f, g = sde.reverse(score_fn).discretize(x, y, t, stepsize)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + g[:, None, None, None] * z
        return x, x_mean


@PredictorRegistry.register('none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, y, t, *args):
        return x, x
