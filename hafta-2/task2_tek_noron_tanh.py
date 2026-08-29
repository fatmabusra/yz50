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
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.data + other.data, (self, other), "+")

        def _backward_add():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward_add
        return result

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.data * other.data, (self, other), "*")

        def _backward_mul():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = _backward_mul
        return result

    def tanh(self):
        t = math.tanh(self.data)
        result = Value(t, (self,), "tanh")

        def _backward_tanh():
           
            self.grad += (1.0 - t ** 2) * result.grad
        result._backward = _backward_tanh
        return result


if __name__ == "__main__":
   
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    bias = Value(6.881373587019543)

    # Forward pass: x1*w1 + x2*w2 + bias
    part1 = x1 * w1
    part2 = x2 * w2
    toplam = part1 + part2
    n = toplam + bias
    out = n.tanh()

    print("Tek nöron çıktısı:", out.data)

    # Manuel zincir kuralı (Chain Rule) çalıştırma
    out.grad = 1.0
    out._backward()
    n._backward()
    toplam._backward()
    part1._backward()
    part2._backward()

    print("\nHesaplanan Gradientler:")
    print("x1.grad:", x1.grad)
    print("w1.grad:", w1.grad)
    print("x2.grad:", x2.grad)
    print("w2.grad:", w2.grad)
    print("bias.grad:", bias.grad)