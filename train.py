import time

import jax
import optax
from datasets import load_dataset
from flax import nnx

from data import CharTokenizer, create_dataloader
from diffusion.scheduler import NoiseScheduler
from loss import loss_func
from model import DiffusionLM
from utils import count_parameters


@nnx.jit
def train_one_batch(batch, noisy_batch, t, model, optimizer):
    loss, grads = nnx.value_and_grad(loss_func, argnums=nnx.DiffState(0, nnx.Param))(
        model, noisy_batch, batch, t
    )
    optimizer.update(model, grads)
    return loss


def train(dataloader, model, optimizer, scheduler, rngs, key, epochs=10):
    print("Starting Training")
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader()):
            print(batch_idx)
            timesteps = jax.random.randint(key, batch.shape[0], 0, scheduler.T)
            noisy_batch = scheduler.noise_sample(batch, timesteps, key)
            loss = train_one_batch(batch, noisy_batch, timesteps, model, optimizer)
            if batch_idx % 10 == 0:
                print(
                    "Loss at Epoch idx : {0} ,Batch idx : {1} : {2}".format(
                        epoch, batch_idx, loss
                    )
                )


if __name__ == "__main__":
    tsd = load_dataset("roneneldan/TinyStories", split="train")
    tokenizer = CharTokenizer()
    dataloader = create_dataloader(
        dataset=tsd, batch_size=64, seq_len=128, tokenizer=tokenizer
    )
    scheduler = NoiseScheduler(vocab_size=128)
    model = DiffusionLM(128, 64, 4, 3, 128, rngs=nnx.Rngs(0))
    print("Total parameters in the model : ", count_parameters(model))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    key = jax.random.key(0)
    train(dataloader, model, optimizer, scheduler, rngs=nnx.Rngs(0), key=key)
