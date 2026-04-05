import os
import math
import random
import pickle
from multiprocessing import Pool, cpu_count

import src.autograd_poly as autograd_poly
import src.polys as polys
from src.microgpt import GPT

num_workers = 25

print(f"Using {num_workers} worker processes.")

random.seed(42)
polys.set_degree(3)

num_steps = 1000
num_samples = 500

# Load the dataset.
docs = [line.strip() for line in open('data/train.txt') if line.strip()]
uchars = sorted(set("".join(docs)))
random.shuffle(docs)


if not os.path.exists('data/fig3'):
    os.mkdir('data/fig3')


def train_one_model(samp):
    print(f"Training model {samp+1:4d} / {num_samples:4d}")

    # Re-initialize the model.
    gpt = GPT(autograd_poly.Value, uchars)
    gpt.load("data/fig3/init.pkl")

    # Sample direction
    direction = [random.gauss(0,1) + random.gauss(0,1) * 1j for _ in range(num_steps)]
    direction_norm = math.sqrt(sum(abs(direction_i)**2 for direction_i in direction))
    direction = [direction_i / direction_norm for direction_i in direction]
    with open(f"data/fig3/direction_{samp}.pkl", "wb") as direction_file:
        pickle.dump(direction, direction_file)

    # Construct downweight polynomials.
    downweights = [polys.Poly([0, direction_i]) for direction_i in direction]

    # Train model.
    gpt.train(docs[:num_steps], downweights=downweights)
    gpt.save(f"data/fig3/model_{samp}.pkl")


if __name__ == "__main__":
    # Initialize the model.
    if not os.path.exists("data/fig3/init.pkl"):
        gpt = GPT(autograd_poly.Value, uchars)
        gpt.save("data/fig3/init.pkl")

    # Push directions through training in parallel.
    with Pool(processes=num_workers) as pool:
        pool.map(train_one_model, range(num_samples))
