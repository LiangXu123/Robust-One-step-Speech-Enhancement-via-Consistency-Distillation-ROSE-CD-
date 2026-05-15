
if noise is None:
    noise = torch.randn_like(x_start)

dims = x_start.ndim


def denoise_fn(x, t):

    # return skip connect result = c_out * model_output + c_skip * x_t
    return self.denoise(model, x, t, **model_kwargs)[1]


if target_model:

    @torch.no_grad()
    def target_denoise_fn(x, t):
        # return skip connect result = c_out * model_output + c_skip * x_t
        return self.denoise(target_model, x, t, **model_kwargs)[1]

else:
    raise NotImplementedError("Must have a target model")

if teacher_model:

    @torch.no_grad()
    def teacher_denoise_fn(x, t):
        # return skip connect result = c_out * model_output + c_skip * x_t
        return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]


@torch.no_grad()
def heun_solver(samples, t, next_t, x0):
    # this is from EDM, Algorithm 2 Our stochastic sampler with σ(t) = t and s(t) = 1
    x = samples
    if teacher_model is None:
        denoiser = x0
    else:
        denoiser = teacher_denoise_fn(x, t)
    # k1 = step_size * f(xt,t)
    d = (x - denoiser) / append_dims(t, dims)
    # yn+1 = yn + k1 * t
    samples = x + d * append_dims(next_t - t, dims)
    if teacher_model is None:
        denoiser = x0
    else:
        denoiser = teacher_denoise_fn(samples, next_t)
    # k2 = step_size * f(xt+1,t+1)
    next_d = (samples - denoiser) / append_dims(next_t, dims)
    # yn+1 = yn + h/2(k1 + k2)
    samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

    return samples


@torch.no_grad()
def euler_solver(samples, t, next_t, x0):
    x = samples
    if teacher_model is None:
        denoiser = x0
    else:
        denoiser = teacher_denoise_fn(x, t)
    d = (x - denoiser) / append_dims(t, dims)
    samples = x + d * append_dims(next_t - t, dims)

    return samples


indices = torch.randint(
    0, num_scales - 1, (x_start.shape[0],), device=x_start.device
)

t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
    self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
)
t = t**self.rho

t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
    self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
)
t2 = t2**self.rho
# t: tensor([9.7232])
# t2: tensor([8.0265])
x_t = x_start + noise * append_dims(t, dims)

dropout_state = torch.get_rng_state()
distiller = denoise_fn(x_t, t)

if teacher_model is None:
    x_t2 = euler_solver(x_t, t, t2, x_start).detach()
else:
    x_t2 = heun_solver(x_t, t, t2, x_start).detach()
_
torch.set_rng_state(dropout_state)
distiller_target = target_denoise_fn(x_t2, t2)
distiller_target = distiller_target.detach()

snrs = self.get_snr(t)
weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
if self.loss_norm == "l1":
    diffs = torch.abs(distiller - distiller_target)
    loss = mean_flat(diffs) * weights
elif self.loss_norm == "l2":
    diffs = (distiller - distiller_target) ** 2
    loss = mean_flat(diffs) * weights
