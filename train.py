import time

import jax
import optax
import orbax.checkpoint as ocp
from datasets import load_dataset
from flax import nnx

import wandb
from data import CharTokenizer, create_dataloader
from diffusion.scheduler import NoiseScheduler
from loss import loss_func
from model import DiffusionLM
from utils import count_parameters

ckpt_dir = ocp.test_utils.erase_and_create_empty(
    "/Users/vaishnavp/Desktop/projects/llm-diffusion/ckpt_dir"
)

wandb.login()


@nnx.jit
def train_one_batch(batch, noisy_batch, t, model, optimizer):
    loss, grads = nnx.value_and_grad(loss_func, argnums=nnx.DiffState(0, nnx.Param))(
        model, noisy_batch, batch, t
    )
    optimizer.update(model, grads)
    return loss


def train(dataloader, model, optimizer, scheduler, checkpointer, rngs, key, epochs=10):
    print("Starting Training")
    total_batches = 0
    with wandb.init(project="diflm", config={"lr": "1e-3"}) as run:
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(dataloader()):
                key, t_key, noise_key = jax.random.split(key, 3)

                timesteps = jax.random.randint(t_key, (batch.shape[0],), 0, scheduler.T)
                noisy_batch = scheduler.noise_sample(batch, timesteps, noise_key)

                loss = train_one_batch(batch, noisy_batch, timesteps, model, optimizer)
                if batch_idx % 10 == 0:
                    if epoch == 0:
                        iter_num = batch_idx
                        total_batches += 10
                    else:
                        iter_num = (epoch * total_batches) + batch_idx
                    run.log({"loss": loss, "iter": iter_num})

                if batch_idx % 100 == 0:
                    print(
                        f"Loss at Epoch idx : {epoch} ,Batch idx : {batch_idx} : {loss}"
                    )
                    _, state = nnx.split(model)
                    checkpointer.save(ckpt_dir / "state", state)


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
    checkpointer = ocp.StandardCheckpointer()
    key = jax.random.key(0)
    train(
        dataloader, model, optimizer, scheduler, checkpointer, rngs=nnx.Rngs(0), key=key
    )
