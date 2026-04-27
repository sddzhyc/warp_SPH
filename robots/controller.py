import torch
from torch import nn


class Controller(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 3
        self.intput_dim = 27
        self.output_dim = 3

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.intput_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim),
        )

        self.register_buffer("action_min", torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.action_min.device)
        control_signal = self.linear_relu_stack(x)
        control_signal = torch.clamp(control_signal, min=self.action_min, max=self.action_max)
        return control_signal


class Controller2D(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.dim = 2 
        # target x 
        # ------------------------------- TODO: add time to state ------------------------------------------------
        self.intput_dim = 12 + 1 # time + 2 parts x 6 (2 for x, 2 for v, 1 for omega, 1 for rotation) 
        # ----------------------------------------------------------------------------------
        # v and omega 
        self.output_dim = 2 # the velocity of body
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.intput_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(), 
                nn.Linear(16, self.output_dim),
                # nn.Softmax(dim=0)
            )
        self.vx_min = -0.5
        self.vx_max = 0.5

        self.vy_min = -0.5
        self.vy_max = 0.
        
        # self.linear_relu_stack.apply(init_weights)
        
    def forward(self, x):
        x = torch.Tensor(x)
        x = x.to(torch.float)
        # control_signal = 2 * (self.linear_relu_stack(x) - 0.5)
        control_signal = self.linear_relu_stack(x) 
        control_signal = torch.clamp(control_signal, 
                         min = torch.Tensor([self.vx_min, self.vy_min]), 
                         max = torch.Tensor([self.vx_max, self.vy_max]))

        return control_signal 
