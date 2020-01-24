from neon.transforms import Transform

class MyReLu(Transform):
    " ReLu activation function. Implements f(x) = max(0,x) "

    def __init__(self, name=None):
        super(MyReLu, self).__init__(name)

    # f(x) = max(0,x)
    def __call__(self, x):
        return self.be.maximum(x, 0)

    # If x > 0, gradient is 1; otherwise 0.
    def bprop(self, x):
        return self.be.greater(x, 0)
