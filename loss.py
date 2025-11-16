import jax
import optax
from flax import linen as nn


def loss_func(params, batch, t, key, model, pad_token=127):
    logits = model.apply(params, batch, t)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
    mask = batch != pad_token
    loss = (loss * mask).sum() / mask.sum()
    return loss
