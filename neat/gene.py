class Gene:

    def __init__(self, in_node, out, weight, enabled: bool, innov):
        self.in_node = in_node
        self.out = out
        self.weight = weight
        self.enabled = enabled
        self.innov = innov

    def __str__(self):
        return str([self.in_node, self.out, self.weight, self.enabled, self.innov])
