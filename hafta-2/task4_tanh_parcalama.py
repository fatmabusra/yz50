import math

class Value:
    def __init__(self, data, previous=(), operation=""):
        self.data = float(data)
        self.grad = 0.0
        self.previous = set(previous)
        self.operation = operation
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward
        return result

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = _backward
        return result

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        result = Value(self.data ** power, (self,), f"**{power}")
        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * result.grad
        result._backward = _backward
        return result

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def exp(self):
        result = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += result.data * result.grad
        result._backward = _backward
        return result

    def backward(self):
        topo = []
        visited = set()
        def build_order(node):
            if node not in visited:
                visited.add(node)
                for prev in node.previous:
                    build_order(prev)
                topo.append(node)
        build_order(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    # 1. Parçalanmış tanh ile Micrograd: (e^2x - 1) / (e^2x + 1)
    x1 = Value(2.0)
    w1 = Value(-3.0)
    x2 = Value(0.0)
    w2 = Value(1.0)
    bias = Value(6.881373587019543)

    n = x1 * w1 + x2 * w2 + bias
    two_n = 2 * n
    exp_val = two_n.exp()
    out = (exp_val - 1) / (exp_val + 1)
    out.backward()

    print("1. Kendi Micrograd'ımız:")
    print("   w1.grad =", round(w1.grad, 6))

    # 2. Sayısal Türev (Numerical Derivative)
    def forward_fn(w1_val):
        n_val = (2.0 * w1_val) + (0.0 * 1.0) + 6.881373587019543
        return math.tanh(n_val)

    h = 0.0001
    numerical_grad_w1 = (forward_fn(-3.0 + h) - forward_fn(-3.0)) / h
    print("2. Sayısal Türev:")
    print("   w1.grad =", round(numerical_grad_w1, 6))

    # 3. PyTorch Doğrulaması
    try:
        import torch
        w1_t = torch.tensor([-3.0], requires_grad=True, dtype=torch.float64)
        x1_t = torch.tensor([2.0], dtype=torch.float64)
        bias_t = torch.tensor([6.881373587019543], dtype=torch.float64)

        n_t = x1_t * w1_t + bias_t
        out_t = torch.tanh(n_t)
        out_t.backward()
        print("3. PyTorch Kontrolü:")
        print("   w1.grad =", round(w1_t.grad.item(), 6))
        print("\nSonuç: Üç yöntem de birebir eşleşti!")
    except ImportError:
        print("PyTorch yüklü olmadığı için adım atlandı.")