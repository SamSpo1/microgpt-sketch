import os
import math
import random
import pickle
import matplotlib.pyplot as plt

import src.autograd_poly as autograd_poly
import src.polys as polys
from src.microgpt import GPT

degree = 3
random.seed(42)
polys.set_degree(degree)

num_steps = 1000
num_samples = 500

# Load the dataset.
docs = [line.strip() for line in open('data/train.txt') if line.strip()]
uchars = sorted(set("".join(docs)))
random.shuffle(docs)

deletions = [i for i, doc in enumerate(docs[:num_steps]) if 'x' in doc]
test_doc = 'max'

### Load the results from Figure 1
with open('results/fig2.pkl', 'rb') as results_file:
        test_losses, _ = pickle.load(results_file)

if not os.path.exists('data/fig3'):
    os.mkdir('data/fig3')

gpt = GPT(autograd_poly.Value, uchars)
gpt.load(f'data/fig3/model_0.pkl')
f_zero = gpt.compute_loss(test_doc).data.coeffs[0].real # type: ignore

### Get the approximations of the Taylor coefficients
taylor_approximations = [0.0 for _ in range(degree)]
for samp in range(num_samples):
    ### Load the direction and model
    with open(f'data/fig3/direction_{samp}.pkl', 'rb') as direction_file:
        direction = pickle.load(direction_file)
    gpt.load(f'data/fig3/model_{samp}.pkl')

    ### Compute <psi, 1_D>
    psi_inner_product = sum(direction[i].conjugate() for i in deletions)

    ### Compute (1/r!) <f^{(r)}(0), psi^{\otimes r}> for r=0,...,degree
    taylor = gpt.compute_loss(test_doc).data.coeffs # type: ignore

    for r in range(1, degree+1):
        taylor_approximations[r-1] += (psi_inner_product ** r * taylor[r] * math.comb(num_steps + r - 1, r)).real / num_samples

### Get the empirical variance of the Taylor coefficients
taylor_variance = [0.0 for _ in range(degree)]
for samp in range(num_samples):
    ### Load the direction and model
    with open(f'data/fig3/direction_{samp}.pkl', 'rb') as direction_file:
        direction = pickle.load(direction_file)
    gpt.load(f'data/fig3/model_{samp}.pkl')

    ### Compute <psi, 1_D>
    psi_inner_product = sum(direction[i].conjugate() for i in deletions)

    ### Compute (1/r!) <f^{(r)}(0), psi^{\otimes r}> for r=0,...,degree
    taylor = gpt.compute_loss(test_doc).data.coeffs # type: ignore

    for r in range(1, degree+1):
        taylor_approx = (psi_inner_product ** r * taylor[r] * math.comb(num_steps + r - 1, r)).real
        taylor_variance[r-1] += (taylor_approx - taylor_approximations[r-1]) ** 2 / num_samples

### Note that var here is just the average variance of a single sample.
### So we need to divide by sqrt(num_samples) to get the variance of the average.
taylor_std = [math.sqrt(var / num_samples) for var in taylor_variance]

xs = [x/100 for x in range(101)]
curves = [[f_zero for _ in xs]]
std_curves = [[0.0 for _ in xs]]
for r in range(1, degree+1):
    curve = [curves[-1][i] + x ** r * taylor_approximations[r-1] for i, x in enumerate(xs)]
    std_curve = [std_curves[-1][i] + x ** r * taylor_std[r-1] for i, x in enumerate(xs)]
    curves.append(curve)
    std_curves.append(std_curve)

with open('results/fig3.pkl', 'wb') as results_file:
    pickle.dump((curves, std_curves), results_file)

wts = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.plot(wts, test_losses, label='empirical', linewidth=6)
for cap, (curve, std_curve) in enumerate(zip(curves[1:], std_curves[1:]), start=1):
    (line,) = plt.plot(xs, curve, label=f'degree {cap}')
    lower = [c - s for c, s in zip(curve, std_curve)]
    upper = [c + s for c, s in zip(curve, std_curve)]
    plt.fill_between(xs, lower, upper, color=line.get_color(), alpha=0.2)
plt.legend()
plt.title(f'Predicted loss curve for {num_samples} directions')
plt.savefig('results/fig3.png')
