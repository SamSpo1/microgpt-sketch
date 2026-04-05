import os
import random
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool

import src.autograd_real as autograd_real
import src.autograd_poly as autograd_poly
import src.polys as polys
from src.microgpt import GPT


random.seed(42)
polys.set_degree(6)

num_steps = 1000
num_workers = 11

# Load the dataset.
docs = [line.strip() for line in open('data/train.txt') if line.strip()]
uchars = sorted(set("".join(docs)))
random.shuffle(docs)
deletion_locs = [i for i, name in enumerate(docs[:num_steps]) if 'x' in name]
print(f'deletion locations: {deletion_locs}')
assert len(deletion_locs) > 0, "must delete something!"
test_doc = 'max'

wts = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]


def train_one_model(args):
    i, wt = args
    print(f"Training model {i+1} / {len(wts)}")

    real_gpt = GPT(autograd_real.Value, uchars)
    real_gpt.load("data/fig2/init.pkl")

    downweights = [(wt if j in deletion_locs else 0) for j in range(num_steps)]
    real_gpt.train(docs[:num_steps], downweights=downweights)
    real_gpt.save(f"data/fig2/model_{wt}.pkl")

    test_loss = real_gpt.compute_loss(test_doc).scalar  # type: ignore
    return i, test_loss


if __name__ == "__main__":
    if not os.path.exists('data/fig2'):
        os.mkdir('data/fig2')

    # Initialize the model parameters.
    real_gpt = GPT(autograd_real.Value, uchars)
    real_gpt.save("data/fig2/init.pkl")

    ### Train a bunch of models with various downweights
    # downweight[i] multiplies the loss computed for training on document i. So:
    #       When downweight[i] = 0, we train on document i as usual.
    #       When downweight[i] = 1, we ignore document i during training.
    # Observe that as we increase the downweight on document 0, the loss of the trained model on document 0 increases.
    with Pool(processes=num_workers) as pool:
        results = pool.map(train_one_model, list(enumerate(wts)))

    results.sort()
    test_losses = [loss for _, loss in results]

    ### Train one model with variable downweight
    downweights = [(polys.Poly([0,1]) if i in deletion_locs else 0) for i in range(num_steps)]

    poly_gpt = GPT(autograd_poly.Value, uchars)
    poly_gpt.load("data/fig2/init.pkl")
    poly_gpt.train(docs[:num_steps], downweights=downweights)
    poly_gpt.save("data/fig2/model_poly.pkl")
    poly_test_loss = poly_gpt.compute_loss(test_doc).data # type: ignore

    if not os.path.exists('results'):
        os.mkdir('results')
    with open('results/fig2.pkl', 'wb') as results_file:
        pickle.dump((test_losses, poly_test_loss), results_file)

    def as_real(x, tol=1e-12):
        z = complex(x)
        if abs(z.imag) > tol:
            raise ValueError(f"Expected near-real value for plotting, got {z}")
        return float(z.real)

    plt.plot(wts, test_losses, label='true loss', linewidth=6)
    for cap in range(1, 1 + polys.get_degree()):
        err = abs(test_losses[-1] - poly_test_loss.eval(1.0, cap=cap))
        plt.plot(wts, [as_real(poly_test_loss.eval(wt, cap=cap)) for wt in wts],
                 label=f'degree-{cap} error = {err : .2e}')
        print(f'degree-{cap} error = {err}')
    plt.legend()
    plt.savefig('results/fig2.png')
