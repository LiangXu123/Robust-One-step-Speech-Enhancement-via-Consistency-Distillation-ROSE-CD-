import torch
import numpy as np


def std(t):
    t = torch.tensor(t)
    theta = 1.5
    sigma_min = 0.05
    sigma_max = 0.5
    logsig = np.log(sigma_max / sigma_min)
    # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in sde()
    sigma_min, theta, logsig = sigma_min, theta, logsig
    # could maybe replace the two torch.exp(... * t) terms here by cached values **t, equ .10
    return torch.sqrt(
        (
            sigma_min**2
            * torch.exp(-2 * theta * t)
            * (torch.exp(2 * (theta + logsig) * t) - 1)
            * logsig
        )
        /
        (theta + logsig)
    )


def _mean(t):
    t = torch.tensor(t)
    theta = 1.5
    exp_interp = torch.exp(-theta * t)
    # return exp_interp * x0 + (1 - exp_interp) * y
    return exp_interp, (1 - exp_interp)


print(_mean(1))
print(std(1))
print(std(0.999))
print(std(0.01))
print(std(0.03))
# tensor(0.3890)
# tensor(0.3881)
# tensor(0.0108)
# tensor(0.0188)


sigma_data = 0.1


def _c_in(t):
    sigma = std(t)
    return (1.0 / torch.sqrt(sigma**2 + sigma_data**2))


def _c_out(t):
    sigma = std(t)
    return ((sigma * sigma_data) / torch.sqrt(sigma_data**2 + sigma**2))


def _c_skip(t):
    sigma = std(t)
    return (sigma_data**2 / (sigma**2 + sigma_data**2))


t_steps = np.array(torch.linspace(1, 0.03, 3))
# for t in np.arange(0, 10, 0.2):
# t/=10
for t in t_steps:
    print('t:', t)
    print('std: {}, input:{}, output:{}, skip:{}'.format(
        std(t), _c_in(t), _c_out(t), _c_skip(t)))


# t: 0.6766666
# std: 0.18426260352134705, input:4.7698774337768555, output:0.08789099752902985, skip:0.2275172919034958
# t: 0.70153844
# std: 0.1952216923236847, input:4.559062480926514, output:0.08900278806686401, skip:0.20785048604011536
# t: 0.72641027
# std: 0.2068144828081131, input:4.353086471557617, output:0.09002812951803207, skip:0.1894935965538025
# t: 0.75128204
# std: 0.2190798968076706, input:4.152417182922363, output:0.09097111970186234, skip:0.17242568731307983
# t: 0.7761538
# std: 0.23205889761447906, input:3.9574458599090576, output:0.09183605015277863, skip:0.1566137969493866


t_steps = np.array(torch.linspace(1, 0.03, 1))
# for t in np.arange(0, 10, 0.2):
# t/=10
for t in t_steps:
    print('t:', t)
    print('std: {}, input:{}, output:{}, skip:{}'.format(
        std(t), _c_in(t), _c_out(t), _c_skip(t)))

t_steps = np.array(torch.linspace(1, 0.03, 2))
# for t in np.arange(0, 10, 0.2):
# t/=10
for t in t_steps:
    print('t:', t)
    print('std: {}, input:{}, output:{}, skip:{}'.format(
        std(t), _c_in(t), _c_out(t), _c_skip(t)))

t_steps = np.array(torch.linspace(1, 0.03, 3))
# for t in np.arange(0, 10, 0.2):
# t/=10
for t in t_steps:
    print('t:', t)
    print('std: {}, input:{}, output:{}, skip:{}'.format(
        std(t), _c_in(t), _c_out(t), _c_skip(t)))


eps = 1e-4
N = 50


def _sigmas_alphas(t):
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


time_steps = torch.linspace(1, eps, N + 1)

# Initial values
time_prev = time_steps[0] * \
    torch.ones(1)
sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = _sigmas_alphas(
    time_prev)


y = 1.0
xt = y
time_steps = torch.linspace(1, eps, N + 1)


def one_step_ode_solver(xt, y, current_t):
    """The SB-ODE sampler function."""
    with torch.no_grad():
        # Initial values
        time_prev = torch.ones(1)  # 1.0
        sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = _sigmas_alphas(
            time_prev)

        for t in time_steps[1:]:
            # Prepare time steps for the whole batch
            time = t * torch.ones(1)

            # Run DNN
            # current_estimate = model(xt, y, time)
            current_estimate = 1.0

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
                # Update state: weighted sum of previous state, current estimate and prior
                xt = weight_prev * xt + weight_estimate * \
                    current_estimate + weight_prior_mean * y
                print("weight_prev: {}".format(weight_prev))
                print("weight_estimate: {}".format(weight_estimate))
                print("weight_prior_mean: {}".format(weight_prior_mean))
                break

            # Save previous values
            time_prev = time
            alpha_prev = alpha_t
            sigma_prev = sigma_t
            sigma_bar_prev = sigma_bart

        return xt


rst = one_step_ode_solver(xt, y, time_steps[10])
rst = one_step_ode_solver(xt, y, time_steps[20])
