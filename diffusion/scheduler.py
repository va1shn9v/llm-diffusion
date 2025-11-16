import jax
import jax.numpy as jnp


class NoiseScheduler:
    def __init__(self, vocab_size, num_steps=10000):
        self.vocab_size = vocab_size
        self.T = num_steps
        self.mask_token = vocab_size - 1
        self.betas = self.get_beta_schedule()

    def get_beta_schedule(self):
        return jnp.linspace(1e-4, 0.02, self.T)

    def noise_sample(self, batch, t, key):
        alpha_bar_t = jnp.cumprod(1 - self.betas[t])
        mask_prob = 1 - alpha_bar_t[..., None]
        mask = jax.random.bernoulli(key, mask_prob, batch.shape)
        batch = jnp.where(mask, self.mask_token, batch)
        return batch
