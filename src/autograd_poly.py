import math
import src.polys as polys

Poly = polys.Poly

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data if isinstance(data, Poly) else Poly(data)      # scalar value of this node calculated during forward pass
        self.grad = 0                                                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children                                       # children of this node in the computation graph
        self._local_grads = local_grads                                 # local derivative of this node w.r.t. its children

    @property
    def scalar(self):
        if isinstance(self.data, Poly):
            return self._real_const(self.data.coeffs[0], "scalar")
        return self._real_const(self.data, "scalar")

    @staticmethod
    def _real_const(x, context: str, tol: float = 1e-12):
        z = complex(x)
        if abs(z.imag) > tol:
            raise ValueError(f"{context} requires a real constant term")
        return float(z.real)

    def __add__(self, other):
        if isinstance(other, Value):
            return Value(self.data + other.data, (self, other), (1, 1))
        else:
            return Value(self.data + other, (self,), (1,))

    def __mul__(self, other):
        if isinstance(other, Value):
            return Value(self.data * other.data, (self, other), (other.data, self.data))
        else:
            return Value(self.data * other, (self,), (other,))

    def exp(self):
        exp_data = self.data.exp()
        return Value(exp_data, (self,), (exp_data,))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(self.data.log(), (self,), (self.data._inverse_series(),))

    def gelu(self):
        x = self.data

        # Jointly compute GELU value and derivative from the same coefficient stream.
        # This avoids doing two separate series expansions.
        degree = len(x.coeffs) - 1
        a0 = self._real_const(x.coeffs[0], "gelu")
        u = x - a0

        phi = math.exp(-0.5 * a0 * a0) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + math.erf(a0 / math.sqrt(2.0)))

        c1 = Phi + a0 * phi
        gelu_data = Poly(a0 * Phi)
        gelu_prime = Poly(c1)

        if degree == 0:
            return Value(gelu_data, (self,), (gelu_prime,))

        # n = 1 term
        u_pow_n = Poly(u)  # u^1
        gelu_data = gelu_data + u_pow_n * c1

        if degree == 1:
            return Value(gelu_data, (self,), (gelu_prime,))

        # n >= 2 terms (same Hermite recurrence as Poly.gelu)
        He_nm2 = 1.0
        He_nm1 = a0
        inv_fact = 1.0

        for n in range(2, degree + 1):
            u_pow_prev = u_pow_n      # u^(n-1)
            u_pow_n = u_pow_n * u     # u^n

            inv_fact /= n
            He_n = a0 * He_nm1 - (n - 1) * He_nm2
            c_n = ((-1.0) ** (n - 1)) * (He_n - He_nm2) * phi * inv_fact

            gelu_data = gelu_data + u_pow_n * c_n
            gelu_prime = gelu_prime + u_pow_prev * (n * c_n)

            He_nm2, He_nm1 = He_nm1, He_n

        return Value(gelu_data, (self,), (gelu_prime,))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v_grad = v.grad
            for child, local_grad in zip(v._children, v._local_grads):
                if local_grad == 1:
                    if child.grad == 0:
                        child.grad = v_grad
                    else:
                        child.grad += v_grad
                elif local_grad == -1:
                    if child.grad == 0:
                        child.grad = -v_grad
                    else:
                        child.grad -= v_grad
                else:
                    if child.grad == 0:
                        child.grad = local_grad * v_grad
                    else:
                        child.grad += local_grad * v_grad
