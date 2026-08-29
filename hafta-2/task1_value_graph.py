
# GÖREV 1: Value Sınıfı ve Computation Graph


class Value:
    def __init__(self, data, previous=(), operation=""):
        self.data = float(data)
        self.grad = 0.0
        self.previous = set(previous)
        self.operation = operation

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.data + other.data, (self, other), "+")
        return result

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.data * other.data, (self, other), "*")
        return result

    def __rmul__(self, other):
        return self * other


if __name__ == "__main__":
    
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b
    e = d + c

    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d = a * b ->", d)
    print("d'nin önceki düğümleri:", d.previous)
    print("e = d + c ->", e)
    print("e'nin önceki düğümleri:", e.previous)