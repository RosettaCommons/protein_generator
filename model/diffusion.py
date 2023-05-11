import enum
import math

import numpy as np
import torch as th


##########################################################################################

#    DIFFUSION CODE BASE FOR PROTEIN SEQUENCE DIFFUSION WAS ADAPTED FROM LM-DIFFUSION    #

                # (https://github.com/XiangLi1999/Diffusion-LM) #
    
##########################################################################################

class GaussianDiffusion_SEQDIFF:
    """
    T = number of timesteps to set up diffuser with
    
    schedule = type of noise schedule to use linear, cosine, gaussian
    
    noise = type of ditribution to sample from; DEFAULT - normal_gaussian
    
    """

    def __init__(self,
                T=1000,
                schedule='sqrt', 
                sample_distribution='normal',
                sample_distribution_gmm_means=[-1.0, 1.0],
                sample_distribution_gmm_variances=[1.0, 1.0],
                F=1,
                ):
        
        # Use float64 for accuracy.
        betas = np.array(get_named_beta_schedule(schedule, T), dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        self.F = F

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        
        # sample_distribution_params
        self.sample_distribution = sample_distribution
        self.sample_distribution_gmm_means = [float(mean) for mean in sample_distribution_gmm_means]
        self.sample_distribution_gmm_variances = [float(variance) for variance in sample_distribution_gmm_variances]
        
        if self.sample_distribution == 'normal':
            self.noise_function = th.randn_like
        else:
            self.noise_function = self.randnmixture_like


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, mask=None, DEVICE=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        
        # noise_function is determined in init depending on type of noise specified
        noise = self.noise_function(x_start)*(self.F**2)
        if DEVICE != None:
            noise = noise.to(DEVICE)

        assert noise.shape == x_start.shape
        x_sample =  (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise)
        
        if mask is not None:
            x_sample[mask]=x_start[mask]
        
        return x_sample

        
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        
        posterior_mean = (_extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)
        
        posterior_log_variance_clipped = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
    
    def randnmixture_like(self, tensor_like, number_normal=3, weights_normal=None):
    
        if self.sample_distribution_gmm_means and self.sample_distribution_gmm_variances:
            assert len(self.sample_distribution_gmm_means) == len(self.sample_distribution_gmm_variances)

        if not weights_normal:
            mix = th.distributions.Categorical(th.ones(len(self.sample_distribution_gmm_means))) #number_normal
        else:
            assert len(weights_normal) == number_normal
            mix = th.distributions.Categorical(weights_normal)
        #comp = torch.distributions.Normal(torch.randn(number_normal), torch.rand(number_normal))
        comp = th.distributions.Normal(th.tensor(self.sample_distribution_gmm_means), th.tensor(self.sample_distribution_gmm_variances))
        #comp = torch.distributions.Normal([-3, 3], [1, 1])
        #comp = torch.distributions.Normal([-3, 0, 3], [1, 1, 1])
        #comp = torch.distributions.Normal([-3, 0, 3], [1, 1, 1])
        gmm = th.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
        return th.tensor([gmm.sample() for _ in range(np.prod(tensor_like.shape))]).reshape(tensor_like.shape)



def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)

    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: 1-np.sqrt(t + 0.0001),)

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
