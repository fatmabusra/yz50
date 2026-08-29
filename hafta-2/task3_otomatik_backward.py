
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
            # Birden fazla yerde kullanılırsa türevler toplanır (+=)
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

    def backward(self):
        topological_order = []
        visited = set()

        def build_order(node):
            if node not in visited:
                visited.add(node)
                for prev in node.previous:
                    build_order(prev)
                topological_order.append(node)

        build_order(self)

        # Çıktının kendisine göre türevi 1.0'dır
        self.grad = 1.0
        for node in reversed(topological_order):
            node._backward()


if __name__ == "__main__":
    
    # Çoklu kullanım testi: a değişkeni iki kez kullanılıyor
    a = Value(3.0)
    b = a + a  # b = 2*a => db/da = 2 olmalı
    b.backward()
    print("b = a + a için a.grad (Beklenen: 2.0):", a.grad)

    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    loss = (x * y) + z
    loss.backward()
    print("\nloss = x*y + z için:")
    print("x.grad:", x.grad)
    print("y.grad:", y.grad)
    print("z.grad:", z.grad)