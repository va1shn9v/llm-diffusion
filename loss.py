import jax
import optax
from flax import linen as nn


def loss_func(model, noisy_batch, batch, t, pad_token=0):
    logits = model(noisy_batch, t)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
    mask = batch != pad_token
    loss = (loss * mask).sum() / mask.sum()
    return loss
