import torch

from ldm.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


class DDIMSampler:
    def __init__(self, model):
        self.model = model

    def make_schedule(self, steps, eta):
        timesteps = make_ddim_timesteps(steps, self.model.num_timesteps)
        alphas = self.model.alphas_cumprod.detach().cpu().numpy()
        sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphas,
            timesteps,
            eta,
        )
        device = self.model.betas.device
        self.timesteps = timesteps
        self.ddim_sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
        self.ddim_alphas = torch.tensor(
            ddim_alphas,
            dtype=torch.float32,
            device=device,
        )
        self.ddim_alphas_prev = torch.tensor(
            ddim_alphas_prev,
            dtype=torch.float32,
            device=device,
        )
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)

    @torch.inference_mode()
    def sample(self, steps, shape, conditioning, eta=0.0):
        self.make_schedule(steps, eta)
        device = self.model.betas.device
        latent = torch.randn(shape, device=device)

        for index, step in reversed(list(enumerate(self.timesteps))):
            timestep = torch.full(
                (shape[0],),
                step,
                device=device,
                dtype=torch.long,
            )
            latent = self.sample_step(latent, conditioning, timestep, index)
        return latent

    def sample_step(self, latent, conditioning, timestep, index):
        predicted_noise = self.model.apply_model(latent, timestep, conditioning)
        batch = latent.shape[0]
        device = latent.device
        alpha = torch.full(
            (batch, 1, 1, 1),
            self.ddim_alphas[index],
            device=device,
        )
        alpha_prev = torch.full(
            (batch, 1, 1, 1),
            self.ddim_alphas_prev[index],
            device=device,
        )
        sigma = torch.full(
            (batch, 1, 1, 1),
            self.ddim_sigmas[index],
            device=device,
        )
        sqrt_one_minus_alpha = torch.full(
            (batch, 1, 1, 1),
            self.ddim_sqrt_one_minus_alphas[index],
            device=device,
        )

        predicted_start = (
            latent - sqrt_one_minus_alpha * predicted_noise
        ) / alpha.sqrt()
        direction = torch.sqrt(1.0 - alpha_prev - sigma.square()) * predicted_noise
        noise = sigma * noise_like(latent.shape, device)
        return alpha_prev.sqrt() * predicted_start + direction + noise
