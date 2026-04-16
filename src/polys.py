### This just implements polynomial algebra.
### Poly takes the place of float.
import math
import numpy as np

DEFAULT_DEGREE = 3
degree = DEFAULT_DEGREE

def get_degree():
    return degree

def set_degree(new_degree: int):
    global degree
    if isinstance(new_degree, bool) or not isinstance(new_degree, int):
        raise TypeError("degree must be an int")
    if new_degree < 0:
        raise ValueError("degree must be >= 0")
    degree = new_degree

class Poly:
    __slots__ = ("coeffs",)

    @staticmethod
    def _from_coeffs(coeffs: np.ndarray):
        p = Poly.__new__(Poly)
        p.coeffs = coeffs
        return p

    @staticmethod
    def _coerce_scalar(x):
        if isinstance(x, np.generic):
            x = x.item()
        return complex(x)

    @staticmethod
    def _require_real_const(a0, context: str, tol: float = 1e-12):
        a0 = complex(a0)
        if abs(a0.imag) > tol:
            raise ValueError(f"{context} requires a real constant term")
        return float(a0.real)

    def __init__(self, data):
        n = degree + 1

        if isinstance(data, Poly):
            self.coeffs = data.coeffs
        elif isinstance(data, (list, tuple, np.ndarray)):
            arr = np.asarray(data)
            if arr.ndim != 1:
                raise TypeError("Poly expects a 1D coefficient sequence")
            arr = np.asarray(arr, dtype=np.complex128)
            out = np.zeros(n, dtype=np.complex128)
            m = min(n, arr.size)
            out[:m] = arr[:m]
            self.coeffs = out
        elif isinstance(data, (float, int, np.floating, np.integer, complex, np.complexfloating)):
            out = np.zeros(n, dtype=np.complex128)
            out[0] = self._coerce_scalar(data)
            self.coeffs = out
        else:
            raise TypeError("Poly expects a Poly, 1D sequence, int, float, complex, or numpy scalar")

    def __repr__(self):
        return " + ".join(f"{coeff} z^{i}" for i, coeff in enumerate(self.coeffs.tolist()))

    def __add__(self, other):
        if isinstance(other, Poly):
            out = self.coeffs + other.coeffs
        else:
            out = self.coeffs.copy()
            out[0] += self._coerce_scalar(other)
        return Poly._from_coeffs(out)
    
    def __mul__(self, other):
        if isinstance(other, Poly):
            out = np.convolve(self.coeffs, other.coeffs)[: self.coeffs.size]
        else:
            out = self.coeffs * self._coerce_scalar(other)
        return Poly._from_coeffs(out)

    def _inverse_series(self):
        b = self.coeffs
        n = b.size
        b0 = b[0]
        if b0 == 0:
            raise ZeroDivisionError("polynomial with zero constant term is not invertible")
        inv = np.zeros(n, dtype=np.complex128)
        inv[0] = 1.0 / b0
        for k in range(1, n):
            inv[k] = -np.dot(b[1:k + 1], inv[k - 1::-1]) / b0
        return Poly._from_coeffs(inv)

    def __pow__(self, other):
        if isinstance(other, (float, np.floating)):
            other = float(other)
        if isinstance(other, float) and other.is_integer():
            other = int(other)

        if isinstance(other, (int, np.integer)):
            other = int(other)
            if other == 0:
                return Poly(1.0)
            if other == 2:
                return self * self
            if other == -1:
                return self._inverse_series()
            if other < 0:
                return (self._inverse_series()) ** (-other)
            result = Poly(1.0)
            base = Poly(self)
            exp = other
            while exp > 0:
                if exp & 1:
                    result = result * base
                base = base * base
                exp >>= 1
            return result

        if isinstance(other, float):
            a = self.coeffs
            n = a.size
            a0 = a[0]
            if a0 == 0:
                raise ZeroDivisionError("non-integer powers need non-zero constant term")
            a0_real = self._require_real_const(a0, "non-integer powers")
            if a0_real < 0:
                raise ValueError("non-integer powers need a positive constant term")
            if other == 0.5:
                y0 = math.sqrt(a0_real)
                y = np.zeros(n, dtype=np.complex128)
                y[0] = y0
                inv_2y0 = 0.5 / y0
                for k in range(1, n):
                    total = np.dot(y[1:k], y[k - 1:0:-1])
                    y[k] = (a[k] - total) * inv_2y0
                return Poly._from_coeffs(y)
            u = (self * (1.0 / a0_real)) - 1.0
            out = Poly(1.0)
            term = Poly(1.0)
            binom = 1.0
            for k in range(1, n):
                term = term * u
                binom *= (other - (k - 1)) / k
                out = out + term * binom
            return out * (a0_real ** other)

        raise TypeError("power must be int or float")

    def log(self):
        a0 = self.coeffs[0]
        a0_real = self._require_real_const(a0, "log")
        if a0_real <= 0:
            raise ValueError("log requires a positive constant term")
        u = (self * (1.0 / a0_real)) - 1.0
        out = Poly(math.log(a0_real))
        term = Poly(u)
        for k in range(1, self.coeffs.size):
            out = out + term * ((1.0 / k) if k % 2 == 1 else (-1.0 / k))
            term = term * u
        return out

    def exp(self):
        a0 = self.coeffs[0]
        u = self - a0
        out = Poly(1.0)
        term = Poly(1.0)
        inv_fact = 1.0
        for k in range(1, self.coeffs.size):
            term = term * u
            inv_fact /= k
            out = out + term * inv_fact
        return out * np.exp(a0)

    def gelu(self):
        degree = self.coeffs.size - 1
        a0 = self._require_real_const(self.coeffs[0], "gelu")
        u = self - a0

        # Standard normal pdf/cdf at a0
        phi = math.exp(-0.5 * a0 * a0) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + math.erf(a0 / math.sqrt(2.0)))

        # Constant term
        out = Poly(a0 * Phi)

        if degree == 0:
            return out

        # Linear term
        term = u
        out = out + term * (Phi + a0 * phi)

        if degree == 1:
            return out

        # Probabilists' Hermite polynomials:
        # He_0(x) = 1, He_1(x) = x,
        # He_n(x) = x He_{n-1}(x) - (n-1) He_{n-2}(x)
        He_nm2 = 1.0       # He_0(a0)
        He_nm1 = a0        # He_1(a0)

        inv_fact = 1.0     # will become 1/n!

        for n in range(2, degree + 1):
            term = term * u
            inv_fact /= n

            He_n = a0 * He_nm1 - (n - 1) * He_nm2

            coeff = ((-1.0) ** (n - 1)) * (He_n - He_nm2) * phi * inv_fact
            out = out + term * coeff

            He_nm2, He_nm1 = He_nm1, He_n

        return out

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):
        if isinstance(other, (float, int, np.floating, np.integer, complex, np.complexfloating)):
            return Poly._from_coeffs(self.coeffs / self._coerce_scalar(other))
        other = other if isinstance(other, Poly) else Poly(other)
        return self * other._inverse_series()

    def __rtruediv__(self, other):
        other = other if isinstance(other, Poly) else Poly(other)
        return other * self._inverse_series()
    
    def eval(self, z, cap=None):
        z = self._coerce_scalar(z)
        if cap is None:
            return np.polynomial.polynomial.polyval(z, self.coeffs)
        cap = int(cap)
        if cap < 0:
            return 0.0
        return np.polynomial.polynomial.polyval(z, self.coeffs[: cap + 1])
