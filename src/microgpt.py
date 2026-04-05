import pickle
import random   # random.seed, random.choices, random.gauss, random.shuffle

class GPT:
    def __init__(self, ValueClass, uchars):
        # Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
        self.uchars = uchars # unique characters in the dataset become token ids 0..n-1
        self.BOS = len(self.uchars) # token id for a special Beginning of Sequence (BOS) token
        self.vocab_size = len(self.uchars) + 1 # total number of unique tokens, +1 is for BOS

        # Initialize the parameters, to store the knowledge of the model
        self.n_layer = 1     # depth of the transformer neural network (number of layers)
        self.n_embd = 16     # width of the network (embedding dimension)
        self.block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
        self.n_head = 4      # number of attention heads
        self.head_dim = self.n_embd // self.n_head # derived dimension of each head
        self.matrix = lambda nout, nin, std=0.08: [[ValueClass(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
        self.state_dict = {'wte': self.matrix(self.vocab_size, self.n_embd), 'wpe': self.matrix(self.block_size, self.n_embd), 'lm_head': self.matrix(self.vocab_size, self.n_embd)}
        for i in range(self.n_layer):
            self.state_dict[f'layer{i}.attn_wq'] = self.matrix(self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.attn_wk'] = self.matrix(self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.attn_wv'] = self.matrix(self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.attn_wo'] = self.matrix(self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.mlp_fc1'] = self.matrix(4 * self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.mlp_fc2'] = self.matrix(self.n_embd, 4 * self.n_embd)
        self.params = [p for mat in self.state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]

        # Let there be Adam, the blessed optimizer and its buffers
        self.learning_rate, self.beta1, self.beta2, self.eps_adam = 0.01, 0.85, 0.99, 1e-3
        # print(f'eps_adam = {self.eps_adam}')
    
    @staticmethod
    def _is_poly_like(val):
        return hasattr(val, 'coeffs')

    @staticmethod
    def _to_real_scalar(val, tol=1e-12):
        if GPT._is_poly_like(val):
            val = val.coeffs[0]
        z = complex(val)
        if abs(z.imag) > tol:
            raise ValueError(f"Expected near-real value when loading real checkpoint parameter, got {z}")
        return float(z.real)

    def set_params(self, param_vals):
        if len(param_vals) != len(self.params):
            raise ValueError(f"Expected {len(self.params)} parameters, got {len(param_vals)}")

        target_is_poly = self._is_poly_like(self.params[0].data)
        poly_type = type(self.params[0].data) if target_is_poly else None

        for p, v in zip(self.params, param_vals):
            if hasattr(v, 'data'):  # legacy checkpoints saved list[Value]
                v = v.data
            if target_is_poly:
                p.data = v if isinstance(v, poly_type) else poly_type(v)
            else:
                p.data = self._to_real_scalar(v)
            p.grad = 0

    def save(self, path):
        param_vals = [p.data for p in self.params]
        with open(path, 'wb') as file:
            pickle.dump(param_vals, file)
        # print(f'params saved to {path}')

    def load(self, path):
        with open(path, 'rb') as file:
            param_vals = pickle.load(file)
        self.set_params(param_vals)
        # print(f'params loaded from {path}')

    # Define the model architecture: a function mapping tokens and parameters to logits over what comes next
    # Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases
    def linear(self, x, w):
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

    def softmax(self, logits):
        max_val = max(val.scalar for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def rmsnorm(self, x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    def gpt(self, token_id, pos_id, keys, values):
        tok_emb = self.state_dict['wte'][token_id] # token embedding
        pos_emb = self.state_dict['wpe'][pos_id] # position embedding
        x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
        x = self.rmsnorm(x) # note: not redundant due to backward pass via the residual connection

        for li in range(self.n_layer):
            # 1) Multi-head Attention block
            x_residual = x
            x = self.rmsnorm(x)
            q = self.linear(x, self.state_dict[f'layer{li}.attn_wq'])
            k = self.linear(x, self.state_dict[f'layer{li}.attn_wk'])
            v = self.linear(x, self.state_dict[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5 for t in range(len(k_h))]
                attn_weights = self.softmax(attn_logits)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(self.head_dim)]
                x_attn.extend(head_out)
            x = self.linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            # 2) MLP block
            x_residual = x
            x = self.rmsnorm(x)
            x = self.linear(x, self.state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.gelu() for xi in x]
            x = self.linear(x, self.state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = self.linear(x, self.state_dict['lm_head'])
        return logits

    def compute_loss(self, doc):
        tokens = [self.BOS] + [self.uchars.index(ch) for ch in doc] + [self.BOS]
        n = min(self.block_size, len(tokens) - 1)

        # Forward the token sequence through the model, building up the computation graph all the way to the loss
        keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = self.gpt(token_id, pos_id, keys, values)
            probs = self.softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        total_loss = sum(losses[1:], losses[0])
        return total_loss / n  # final average loss over the document sequence. May yours be low.

    def train(self, docs, num_steps=None, downweights=None):
        # Initialize the model and optimizer
        num_docs = len(docs)
        if num_steps is None: num_steps = num_docs
        m = [0.0] * len(self.params) # momentum buffer
        v = [0.0] * len(self.params) # second moment buffer

        # Repeat in sequence
        for step in range(num_steps):
            # Take single document, tokenize it, surround it with BOS special token on both sides
            doc = docs[step % num_docs]
            loss = self.compute_loss(doc)
            if downweights is None:
                reweighted_loss = loss
            else:
                reweighted_loss = loss * (1 - downweights[step % num_docs])

            # Backward the loss, calculating the gradients with respect to all model parameters
            reweighted_loss.backward() # type: ignore

            # Momentum-SGD update: update the model parameters based on the corresponding gradients
            lr_t = self.learning_rate * (1 - step / num_steps) # linear learning rate decay
            for i, p in enumerate(self.params):
                m[i] = self.beta1 * m[i] + (1 - self.beta1) * p.grad
                v[i] = self.beta2 * v[i] + (1 - self.beta2) * p.grad ** 2
                m_hat = m[i] / (1 - self.beta1 ** (step + 1))
                v_hat = v[i] / (1 - self.beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat + self.eps_adam) ** 0.5
                p.grad = 0

            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.scalar:.4f}", end='\r') # type: ignore
        print()

    # Inference: may the model babble back to us
    def sample(self, num_samples, temperature=0.5):
        print("\n--- inference (new, hallucinated names) ---")
        for sample_idx in range(num_samples):
            keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
            token_id = self.BOS
            sample = []
            for pos_id in range(self.block_size):
                logits = self.gpt(token_id, pos_id, keys, values)
                probs = self.softmax([l / temperature for l in logits])
                token_id = random.choices(range(self.vocab_size), weights=[p.scalar for p in probs])[0]
                if token_id == self.BOS:
                    break
                sample.append(self.uchars[token_id])
            print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
