import torch
from torch import nn


class _MLPController(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, action_min, action_max):
        super().__init__()
        self.dim = int(output_dim)
        self.intput_dim = int(input_dim)
        self.output_dim = int(output_dim)

        layers = []
        last_dim = self.intput_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, int(h)))
            layers.append(nn.ReLU())
            last_dim = int(h)
        layers.append(nn.Linear(last_dim, self.output_dim))
        self.linear_relu_stack = nn.Sequential(*layers)

        self.register_buffer("action_min", torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(action_max, dtype=torch.float32))

    def forward(self, x):
        # Keep dtype/device consistent with the model while supporting list/numpy inputs.
        ref = self.action_min
        x = torch.as_tensor(x, dtype=torch.float32, device=ref.device)
        control_signal = self.linear_relu_stack(x)
        control_signal = torch.clamp(control_signal, min=self.action_min, max=self.action_max)
        return control_signal


class Controller(_MLPController):
    def __init__(self) -> None:
        # 3D default version:
        # input: [time + 2 parts state], 27 dims with full quaternion per body.
        # output: [vx, vy, vz].
        super().__init__(
            input_dim=27,
            output_dim=3,
            hidden_dims=(64, 64, 32),
            action_min=[-0.5, -0.5, -0.5],
            action_max=[0.5, 0.0, 0.5],
        )


class Controller2D(_MLPController):
    def __init__(self) -> None:
        # Keep the previous 2D cartpole controller behavior.
        super().__init__(
            input_dim=13,
            output_dim=2,
            hidden_dims=(16, 16),
            action_min=[-0.5, -0.5],
            action_max=[0.5, 0.0],
        )

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
