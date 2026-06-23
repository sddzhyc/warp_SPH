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


class ControllerWithFluidGRU(ControllerWithFluid):
    """
    ControllerWithFluid with a GRU memory unit inserted between the
    concatenated input and the MLP.  The GRU hidden state persists across
    control steps within an episode and is **detached** after every forward
    pass to implement truncated BPTT (k=1).  This prevents "backward through
    the graph a second time" errors in differentiable-physics training loops
    while still allowing the GRU to learn useful temporal features.

    Architecture
    ------------
    state + ae_feat ──→ Linear(128) + ReLU ──→ GRUCell(hidden=128) ──→ MLP ──→ action
    """

    def __init__(self, state_dim: int, ae_feat_dim: int, action_dim: int,
                 hidden=200, has_continuous_action_space=True, action_std_init=0.1,
                 action_min=None, action_max=None):
        super().__init__(state_dim, ae_feat_dim, action_dim,
                         hidden=hidden,
                         has_continuous_action_space=has_continuous_action_space,
                         action_std_init=action_std_init,
                         action_min=action_min,
                         action_max=action_max)

        self.gru_hidden_size = 128

        # 输入编码层：将拼接特征投影到 GRU 输入空间
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
        )

        # 单步 GRU 单元
        self.gru_cell = nn.GRUCell(input_size=128, hidden_size=self.gru_hidden_size)

        # 替换 MLP：输入从 gru_hidden_size 开始（而非原始 input_dim）
        mlp_input_dim = self.gru_hidden_size
        self.net = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )

        self._gru_hidden = None  # 每 episode 重置

    def reset_hidden(self):
        """重置 GRU 隐藏状态（每个新 episode 开始时调用）"""
        self._gru_hidden = None

    def forward(self, state, ae_feat, step=None):
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

        # 编码 → GRU
        encoded = self.input_encoder(x)
        if self._gru_hidden is None:
            self._gru_hidden = torch.zeros(self.gru_hidden_size, device=target_device)
        h = self.gru_cell(encoded, self._gru_hidden)
        # ★ truncated BPTT (k=1): detach 隐状态，防止计算图跨 backward 窗口累积
        self._gru_hidden = h.detach()

        out = self.net(h)
        out = torch.clamp(out, min=self.action_min, max=self.action_max)
        return out

    def forward_batch(self, states, ae_feats):
        # 批量模式下逐样本通过 GRU（不共享隐状态）
        target_device = self.action_min.device
        s = torch.as_tensor(states, dtype=torch.float32, device=target_device)
        a = torch.as_tensor(ae_feats, dtype=torch.float32, device=target_device)
        if a.dim() > 2:
            a = a.view(a.size(0), -1)
        x = torch.cat([s, a], dim=1)

        outs = []
        for i in range(x.size(0)):
            encoded = self.input_encoder(x[i])
            h = self.gru_cell(encoded, torch.zeros(self.gru_hidden_size, device=target_device))
            out_i = self.net(h)
            outs.append(out_i)
        out = torch.stack(outs, dim=0)
        out = torch.max(torch.min(out, self.action_max), self.action_min)
        return out


if __name__ == '__main__':
    # quick smoke test
    c = ControllerWithFluid(state_dim=12, ae_feat_dim=64, action_dim=1)
    s = torch.randn(12)
    f = torch.randn(64)
    print(c(s, f))