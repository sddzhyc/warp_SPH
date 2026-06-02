import torch
from torch import nn

class ControllerWithFluid(nn.Module):
    """
    Controller that takes the original state vector and a flattened/pooled
    autoencoder feature tensor and concatenates them before MLP processing.

    Usage: provide `state_dim` and `ae_feat_dim` (flattened AE encoding size).
    """
    def __init__(self, state_dim:int, ae_feat_dim:int, action_dim:int, hidden=200, has_continuous_action_space=True, action_std_init=0.1, action_min=None, action_max=None):
        super().__init__()
        self.state_dim = state_dim
        self.ae_feat_dim = ae_feat_dim
        self.input_dim = state_dim + ae_feat_dim
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

        # simple MLP
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        )

        # action bounds (default for cartpole-on-water: vx, vy, vz)
        if action_min is None or action_max is None:
            if action_dim == 2:
                action_min = [-0.5, -0.5]
                action_max = [0.5, 0.0]
            elif action_dim == 3:
                action_min = [-0.5, -0.5, -0.5]
                action_max = [0.5, 0.0, 0.5]
            else:
                action_min = [-0.5] * action_dim
                action_max = [0.5] * action_dim
        self.register_buffer('action_min', torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer('action_max', torch.tensor(action_max, dtype=torch.float32))

    def forward(self, state, ae_feat, step=None):
        # state: tensor or numpy array of shape (state_dim,)
        # ae_feat: tensor of shape (ae_feat_dim,) or feature map which will be flattened
        target_device = self.action_min.device
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=target_device)
        else:
            state = state.to(dtype=torch.float32, device=target_device)
        if not torch.is_tensor(ae_feat):
            ae_feat = torch.as_tensor(ae_feat, dtype=torch.float32, device=target_device)
        else:
            ae_feat = ae_feat.to(dtype=torch.float32, device=target_device)

        ae_flat = ae_feat.view(-1) if ae_feat.dim() > 1 else ae_feat
        x = torch.cat([state.view(-1), ae_flat.view(-1)], dim=0)
        out = self.net(x)
        out = torch.clamp(out, min=self.action_min, max=self.action_max)
        return out

    def forward_batch(self, states, ae_feats):
        # batch-aware forward: states (B, S), ae_feats (B, F) or feature maps
        target_device = self.action_min.device
        s = torch.as_tensor(states, dtype=torch.float32, device=target_device)
        a = torch.as_tensor(ae_feats, dtype=torch.float32, device=target_device)
        if a.dim() > 2:
            a = a.view(a.size(0), -1)
        x = torch.cat([s, a], dim=1)
        out = self.net(x)
        out = torch.max(torch.min(out, self.action_max), self.action_min)
        return out

if __name__ == '__main__':
    # quick smoke test
    c = ControllerWithFluid(state_dim=12, ae_feat_dim=64, action_dim=2)
    s = torch.randn(12)
    f = torch.randn(64)
    print(c(s, f))