import jax
import jax.numpy as jnp
from flax import nnx

from data import CharTokenizer
from diffusion.scheduler import NoiseScheduler
from model import DiffusionLM
from utils import generate_text

model = DiffusionLM(128, 64, 4, 3, 128, rngs=nnx.Rngs(0))
tokenizer = CharTokenizer()
scheduler = NoiseScheduler(vocab_size=128)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
user_prompt = input("Enter the User Prompt : ")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

print("Generating Output : \n")
print(" ", end="")

generate_text(model, user_prompt, length=128, tokenizer=tokenizer, scheduler=scheduler)
