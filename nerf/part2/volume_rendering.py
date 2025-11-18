import torch


def volrend(sigmas, rgbs, step_size):
    #sigmas:(N_rays, N_samples, 1)
    #rgbs: (N_rays, N_samples, 3)
    #step_size: scalar
    #returns: (N_rays, 3)
    # Compute alpha (opacity)
    alpha = 1.0 - torch.exp(-sigmas * step_size) #alpha used to calculate opacity of each point
    
    #(probability of not being occluded by any other points)
    cum_sigma = torch.cumsum(sigmas * step_size, dim=1) #cumulative sigma (sum of sigma values along the ray)
    cum_sigma_shifted = torch.cat([torch.zeros_like(cum_sigma[:, :1]), cum_sigma[:, :-1]], dim = 1) #shifted cumulative sigma
    T = torch.exp(-cum_sigma_shifted) #transmittance
    #weights used to composite colors
    weights = T * alpha #weights used to composite colors
    rgb_map = torch.sum(weights * rgbs, dim = 1) #composite colors
    return rgb_map #(N_rays, 3) - one color per ray

torch.manual_seed(42)
sigmas = torch.rand((10, 64, 1))
rgbs = torch.rand((10, 64, 3))
step_size = (6.0 - 2.0) / 64
rendered_colors = volrend(sigmas, rgbs, step_size)

correct = torch.tensor([
    [0.5006, 0.3728, 0.4728],
    [0.4322, 0.3559, 0.4134],
    [0.4027, 0.4394, 0.4610],
    [0.4514, 0.3829, 0.4196],
    [0.4002, 0.4599, 0.4103],
    [0.4471, 0.4044, 0.4069],
    [0.4285, 0.4072, 0.3777],
    [0.4152, 0.4190, 0.4361],
    [0.4051, 0.3651, 0.3969],
    [0.3253, 0.3587, 0.4215]
  ])

result = torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)
print(result)
assert result