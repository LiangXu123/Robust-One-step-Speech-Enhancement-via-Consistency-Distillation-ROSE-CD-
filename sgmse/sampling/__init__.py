# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch
import numpy as np
from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_euler_sampler(
    predictor_name, sde, score_fn,
        denoise=True, eps=3e-2, probability_flow: bool = True
):
    predictor_cls = PredictorRegistry.get_by_name(
        predictor_name)
    predictor = predictor_cls(
        sde, score_fn, probability_flow=probability_flow)  # false

    def sampler_at_one_step(xt, y, t, next_t):
        """The PC sampler function. here for distill, we only backward for once"""
        # print('denoise: {}'.format(denoise))
        # denoise: True
        with torch.no_grad():
            stepsize = t - next_t
            vec_t = torch.ones(y.shape[0], device=y.device) * t
            xt, xt_mean = predictor.update_fn(xt, y, vec_t)
            if denoise == True:
                return xt_mean
        return xt

    return sampler_at_one_step


def get_heun_solver_sampler(
    predictor_name, sde, score_fn,
        denoise=True, eps=3e-2, probability_flow: bool = False
):
    predictor_cls = PredictorRegistry.get_by_name(
        predictor_name)
    predictor = predictor_cls(
        sde, score_fn, probability_flow=probability_flow)  # false

    def sampler_at_one_step(xt, y, t, next_t):
        """The PC sampler function. here for distill, we only backward for once"""
        # print('denoise: {}'.format(denoise))
        # denoise: True
        with torch.no_grad():
            stepsize = t - next_t
            vec_t = torch.ones(y.shape[0], device=y.device) * t
            xt, xt_mean = predictor.update_fn(xt, y, vec_t)
            if denoise == True:
                return xt_mean
        return xt

    return sampler_at_one_step


def get_mix_sampler(
    predictor_name, sde, score_fn,
        denoise=True, eps=3e-2, probability_flow: bool = True
):
    predictor_cls = PredictorRegistry.get_by_name(
        predictor_name)
    predictor = predictor_cls(
        sde, score_fn, probability_flow=probability_flow)  # false

    def sampler_at_one_step(xt, y, t, next_t):
        """The mix sampler function. here for distill, we only backward for once"""
        with torch.no_grad():
            stepsize = t - next_t
            vec_t = torch.ones(y.shape[0], device=y.device) * t
            xt, xt_mean = predictor.update_fn(xt, y, vec_t)
            if denoise == True:
                return xt_mean
        return xt

    return sampler_at_one_step


def _sigmas_alphas(t, eps=0.03):
    c = 0.4
    k = 2.6
    alpha_t = torch.ones_like(t)
    alpha_T = torch.ones_like(t)
    sigma_t = torch.sqrt((c*(k**(2*t)-1.0))
                         # Table 1
                         / (2*torch.log(torch.tensor(k))))
    sigma_T = torch.sqrt((c*(k**(2*1)-1.0))
                         # Table 1
                         / (2*torch.log(torch.tensor(k))))

    # below Eq. (9)
    alpha_bart = alpha_t / (alpha_T + eps)
    sigma_bart = torch.sqrt(sigma_T**2 - sigma_t **
                            2 + eps)             # below Eq. (9)

    return sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart


def get_sb_solver(
    sde, score_fn,
        denoise=True, eps=3e-2
):
    def sb_sampler_at_one_step(xt, y, current_t, next_t):
        """The sb ODE sampler function. here for distill, we only backward for once"""
        # print('denoise: {}'.format(denoise))
        # denoise: True
        time_steps = torch.linspace(sde.T, eps, sde.N)
        with torch.no_grad():
            # Initial values
            time_prev = torch.ones(1)  # 1.0
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = _sigmas_alphas(
                time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(1)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = _sigmas_alphas(
                    time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t * sigma_bart / \
                    (alpha_prev * sigma_prev * sigma_bar_prev + eps)
                weight_estimate = (
                    alpha_t
                    / (sigma_T**2 + eps)
                    * (sigma_bart**2 - sigma_bar_prev * sigma_t * sigma_bart / (sigma_prev + eps))
                )
                weight_prior_mean = (
                    alpha_t
                    / (alpha_T * sigma_T**2 + eps)
                    * (sigma_t**2 - sigma_prev * sigma_t * sigma_bart / (sigma_bar_prev + eps))
                )

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_prior_mean = weight_prior_mean[:, None, None, None]

                if t == current_t:
                    # Run Score DNN
                    # current_estimate = model(xt, y, time)
                    # grad = self.score_fn(x, y, t, *args)
                    # print("weight_prev: {}".format(weight_prev))
                    # print("weight_estimate: {}".format(weight_estimate))
                    # print("weight_prior_mean: {}".format(weight_prior_mean))
                    weight_prev = weight_prev.to(y.device)
                    weight_estimate = weight_estimate.to(y.device)
                    weight_prior_mean = weight_prior_mean.to(y.device)
                    vec_t = torch.ones(y.shape[0], device=y.device) * t
                    current_estimate = score_fn(xt, y, vec_t)
                    # Update state: weighted sum of previous state, current estimate and prior
                    xt = weight_prev * xt + weight_estimate * \
                        current_estimate + weight_prior_mean * y
                    break

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt
    return sb_sampler_at_one_step


def get_sb_solver_sde(sde, score_fn, denoise=True, eps=3e-2):
    def sb_sampler_at_one_step(xt, y, current_t, next_t):
        """The SB ODE-like sampler function (single-step backward)."""
        time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

        with torch.no_grad():
            # Initial values
            time_prev = time_steps[0] * \
                torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = _sigmas_alphas(
                time_prev)

            for t in time_steps[1:]:
                time = t * torch.ones(xt.shape[0], device=xt.device)
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = _sigmas_alphas(
                    time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t**2 / \
                    (alpha_prev * sigma_prev**2 + eps)
                tmp = 1 - sigma_t**2 / (sigma_prev**2 + eps)
                weight_estimate = alpha_t * tmp
                weight_z = alpha_t * sigma_t * torch.sqrt(tmp)

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_z = weight_z[:, None, None, None]

                # Only apply update at current_t
                if t == current_t:
                    # Noise is only injected if desired (but usually for distillation, denoise=True)
                    z_norm = torch.randn_like(xt) if not denoise else 0.0
                    # Run DNN
                    weight_prev = weight_prev.to(y.device)
                    weight_estimate = weight_estimate.to(y.device)
                    weight_prior_mean = weight_prior_mean.to(y.device)
                    vec_t = torch.ones(y.shape[0], device=y.device) * t
                    current_estimate = score_fn(xt, y, vec_t)
                    xt = weight_prev * xt + weight_estimate * current_estimate + weight_z * z_norm
                    break

                # Update for next iteration
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt
    return sb_sampler_at_one_step


def get_pc_sampler2(
    predictor_name, corrector_name, sde, score_fn,
        denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        xt: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(
        predictor_name)  # reverse_diffusion
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)  # ald
    predictor = predictor_cls(
        sde, score_fn, probability_flow=probability_flow)  # false
    corrector = corrector_cls(sde, score_fn, snr=snr,  # 0.5
                              n_steps=corrector_steps)  # 1

    def pc_sampler_at_one_step(xt, y, t, next_t):
        """The PC sampler function. here for distill, we only backward for once"""
        # print('denoise: {}'.format(denoise))
        # denoise: True
        with torch.no_grad():
            stepsize = t - next_t
            vec_t = torch.ones(y.shape[0], device=y.device) * t

            # this is different from Predictor-Corrector pipeline !!!!!!!
            xt, xt_mean = corrector.update_fn(xt, y, vec_t)
            # corrector="ald"
            # predictor="reverse_diffusion"
            # if denoise == True:
            # xt = xt_mean
            # in distill2 ,we comment above two lines
            xt, xt_mean = predictor.update_fn(xt, y, vec_t, stepsize)

            # xt, xt_mean = corrector.update_fn(xt, y, vec_t)
            if denoise == True:
                return xt_mean
        return xt

    return pc_sampler_at_one_step





def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, y,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(
        predictor_name)  # reverse_diffusion
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)  # ald
    predictor = predictor_cls(
        sde, score_fn, probability_flow=probability_flow)  # false
    corrector = corrector_cls(sde, score_fn, snr=snr,  # 0.5
                              n_steps=corrector_steps)  # 1
    # print('denoise: {}'.format(denoise))
    # denoise: True

    def pc_sampler():
        """The PC sampler function."""
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)
            # xt = y + z*sigma
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            for i in range(sde.N):  # 40
                t = timesteps[i]
                # tensor(40)
                # tensor(38.9751)
                # ...
                # tensor(2.0797)
                # tensor(1.0549)
                # tensor(0.0300)
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1]  # from eps to 0
                vec_t = torch.ones(y.shape[0], device=y.device) * t

                # this is different from Predictor-Corrector pipeline !!!!!!!
                xt, xt_mean = corrector.update_fn(xt, y, vec_t)
                # print('xt.shape:{}, y.shape:{}, xt_mean.shape:{}'.format
                #   (xt.shape, y.shape, xt_mean.shape))
                #   xt.shape:torch.Size([1, 1, 256, 256]), y.shape:torch.Size([1, 1, 256, 256]), xt_mean.shape:torch.Size([1, 1, 256, 256])
                # corrector="ald"
                # predictor="reverse_diffusion"
                xt, xt_mean = predictor.update_fn(xt, y, vec_t, stepsize)

                # xt, xt_mean = corrector.update_fn(xt, y, vec_t)

            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)
            return x_result, ns

    return pc_sampler


def get_ode_sampler(
    sde, score_fn, y, inverse_scaler=None,
    denoise=True, rtol=1e-5, atol=1e-5,
    method='RK45', eps=3e-2, device='cuda', **kwargs
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(
        sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor.update_fn(x, y, vec_eps)
        return x

    def drift_fn(x, y, t):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, y, t)[0]

    def ode_sampler(z=None, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = sde.prior_sampling(y.shape, y).to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(
                    device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, y, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method, **kwargs
            )
            nfe = solution.nfev
            x = torch.tensor(
                solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler


def get_sb_sampler(sde, model, y, eps=1e-4, n_steps=50, sampler_type="ode", **kwargs):
    # adapted from https://github.com/NVIDIA/NeMo/blob/78357ae99ff2cf9f179f53fbcb02c88a5a67defb/nemo/collections/audio/parts/submodules/schroedinger_bridge.py#L382
    def sde_sampler():
        """The SB-SDE sampler function."""
        with torch.no_grad():
            xt = y[:, [0], :, :]  # special case for storm_2ch
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * \
                torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(
                time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(
                    time)

                # Run DNN
                current_estimate = model(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t**2 / \
                    (alpha_prev * sigma_prev**2 + sde.eps)
                tmp = 1 - sigma_t**2 / (sigma_prev**2 + sde.eps)
                weight_estimate = alpha_t * tmp
                weight_z = alpha_t * sigma_t * torch.sqrt(tmp)

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_z = weight_z[:, None, None, None]

                # Random sample
                z_norm = torch.randn_like(xt)

                if t == time_steps[-1]:
                    weight_z = 0.0

                # Update state: weighted sum of previous state, current estimate and noise
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_z * z_norm

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, n_steps

    def ode_sampler():
        """The SB-ODE sampler function."""
        with torch.no_grad():
            xt = y
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * \
                torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(
                time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(
                    time)

                # Run DNN
                current_estimate = model(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t * sigma_bart / \
                    (alpha_prev * sigma_prev * sigma_bar_prev + sde.eps)
                weight_estimate = (
                    alpha_t
                    / (sigma_T**2 + sde.eps)
                    * (sigma_bart**2 - sigma_bar_prev * sigma_t * sigma_bart / (sigma_prev + sde.eps))
                )
                weight_prior_mean = (
                    alpha_t
                    / (alpha_T * sigma_T**2 + sde.eps)
                    * (sigma_t**2 - sigma_prev * sigma_t * sigma_bart / (sigma_bar_prev + sde.eps))
                )

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_prior_mean = weight_prior_mean[:, None, None, None]

                # Update state: weighted sum of previous state, current estimate and prior
                xt = weight_prev * xt + weight_estimate * \
                    current_estimate + weight_prior_mean * y

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, n_steps

    if sampler_type == "sde":
        return sde_sampler
    elif sampler_type == "ode":
        return ode_sampler
    else:
        raise ValueError("Invalid type. Choose 'ode' or 'sde'.")


def get_stochastic_sampler(score_fn, sde, y, snr=0.5, eps=3e-2):
    def stochastic_sampler():
        with torch.no_grad():
            # Initial noise x_T
            xt = sde.prior_sampling(y.shape, y).to(y.device)

            # Initial consistency model call: x ← f_θ(x_T, T)
            vec_T = torch.ones(y.shape[0], device=y.device) * sde.T
            x = score_fn(xt, y, vec_T, return_X=True)

            if sde.N == 1:
                return x, x
            else:
                # Multistep sampling
                N = sde.N
                timesteps = torch.linspace(sde.T, eps, N, device=y.device)

                # for n = 1 to N-1 do
                for n in range(1, N):
                    tau_n = timesteps[n]  # τₙ

                    # Sample z ~ N(0,I)
                    z = torch.randn_like(x)

                    # x̂_τₙ ← x + √(τₙ² - ε²)z
                    # tau_n_tensor = tau_n.clone().detach().to(device=x.device, dtype=x.dtype)
                    # eps_tensor = torch.tensor(eps, device=x.device, dtype=x.dtype)

                    # coeff = torch.sqrt(tau_n_tensor ** 2 - eps_tensor ** 2)
                    # print("coeff:", coeff)
                    # x_hat_tau_n = x + coeff * z

                    vec_t = torch.ones(y.shape[0], device=y.device) * tau_n
                    std = sde.marginal_prob(xt, y, vec_t)[1]  # std=sigma
                    # x_mean = score_fn(xt, y, vec_t, return_X=True)
                    step_size = (snr * std) ** 2 * 2  # this is theta
                    # print("coeff:", torch.sqrt(step_size * 2))
                    x_hat_tau_n = x + z * \
                        torch.sqrt(step_size * 2)[:, None, None, None]

                    # x ← f_θ(x̂_τₙ, τₙ)
                    vec_tau_n = torch.ones(y.shape[0], device=y.device) * tau_n
                    x = score_fn(x_hat_tau_n, y, vec_tau_n, return_X=True)

                return x, x

    return stochastic_sampler


# @torch.no_grad()
# def stochastic_iterative_sampler(
#     distiller,
#     x,
#     sigmas,
#     generator,
#     ts,
#     progress=False,
#     callback=None,
#     t_min=0.002,
#     t_max=80.0,
#     rho=7.0,
#     steps=40,
# ):
#     t_max_rho = t_max ** (1 / rho)
#     t_min_rho = t_min ** (1 / rho)
#     s_in = x.new_ones([x.shape[0]])

#     # ts = 0, 22, 39
#     for i in range(len(ts) - 1):
#         t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
#         x0 = distiller(x, t * s_in) # this is the skip connection output, e.g. Denoiser output
#         next_t = (t_max_rho + ts[i + 1] / (steps - 1)
#                   * (t_min_rho - t_max_rho)) ** rho
#         next_t = np.clip(next_t, t_min, t_max)
#         # the last next_t is 0.00200
#         # the last np.sqrt(next_t**2 - t_min**2)=1.1271865066977384e-10, so the last predict is x0
#         x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

#     return x
