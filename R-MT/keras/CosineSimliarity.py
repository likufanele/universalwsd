class CosineCorrelation(Cost):
    def __init__(self, scale=1):
        """
        Args:
            scale (float, optional): Amount by which to scale the backpropagated error (default: 1)
        """
        self.scale = scale

    def __call__(self, y, t):
        """
        Returns the binary cross entropy cost.

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the binary cross entropy cost
        """
        assert y.shape == t.shape, "CrossEntropy requires network output shape to match targets"
        return self.be.sum(self.be.safelog(1 - y) * (t - 1) - self.be.safelog(y) * t, axis=0)

    
    def bprop(self, y, t):
        """
        Returns the derivative of the binary cross entropy cost.

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the (mean) shortcut derivative of the binary entropy
                    cost function ``(y - t) / y.shape[1]``
        """
        return self.scale * (y - t)
