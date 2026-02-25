import warp as wp

class Task:
    def __init__(self, sim):
        self.sim = sim

    def init_targets(self):
        pass

    def init_optimizer(self):
        pass

    def compute_loss(self):
        pass

    def init_simulation_state(self, t):
        pass

    def clear_grad(self):
        pass