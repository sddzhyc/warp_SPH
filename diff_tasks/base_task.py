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

    def init_simulation_state(self):
        pass

    def clear_grad(self):
        pass

    def get_loss_state_info(self):
        return {}

    def norm_final_grad(self, v_grad, materialMarks):
        pass