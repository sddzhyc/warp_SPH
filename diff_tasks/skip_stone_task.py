import warp as wp
import numpy as np
from norm_grad_utils import norm_grad_vec3
from .base_task import Task


@wp.kernel
def assign_rigid_init_state_kernel(
    rigid_v: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
    opt_rigid_v: wp.array(dtype=wp.vec3),
    opt_rigid_omega: wp.array(dtype=wp.vec3),
    rigid_id: int,
):
    tid = wp.tid()
    if tid == 0:
        rigid_v[rigid_id] = opt_rigid_v[0]
        rigid_omega[rigid_id] = opt_rigid_omega[0]

@wp.kernel
def compute_rigid_loss(
    rigid_x: wp.array(dtype=wp.vec3),
    target_rigid_x: wp.array(dtype=wp.vec3),
    rigid_vel: wp.array(dtype=wp.vec3),
    opt_rigid_vel: wp.array(dtype=wp.vec3),
    stone_rigid_id: int,
    rigid_q: wp.array(dtype=wp.quat),
    target_rigid_q: wp.array(dtype=wp.quat),
    loss: wp.array(dtype=float)
):
    tid = wp.tid()
    upperbound = -5.0
    if tid == 0: # 排除rigid body0（流体块）
        return
    # Position loss per rigid body
    diff_pos = norm_grad_vec3(rigid_x[tid] - target_rigid_x[tid])
    l_pos = wp.dot(diff_pos, diff_pos)
    
    w_penalty = 1.0
    loss_penalty = 0.0
    if tid == stone_rigid_id:
        loss_penalty = w_penalty * wp.max(opt_rigid_vel[0][1] - upperbound, 0.0)
    # Rotation loss (0.5 * ||q - tq||^2)
    # q = rigid_q[tid]
    # tq = target_rigid_q[tid]
    # wp.printf("Rigid body %d: q = (%.9f, %.9f, %.9f, %.9f), tq = (%.9f, %.9f, %.9f, %.9f)\n", tid, q.x, q.y, q.z, q.w, tq.x, tq.y, tq.z, tq.w)
    # l_rot = 0.5 * (wp.dot(q, q) + wp.dot(tq, tq) - 2.0 * wp.dot(q, tq))
    w_pos = 1.

    # Combine losses (add weights here if needed)
    total_loss = w_pos * l_pos + loss_penalty
    # wp.printf("Rigid body %d: Rotation loss = %.9e, Position loss = %.9e, Total loss = %.9e\n", tid, l_rot, l_pos, total_loss)

    wp.atomic_add(loss, 0, total_loss)


class SkipStoneTask(Task):
    def __init__(self, sim):
        super().__init__(sim)
        self.stone_rigid_id = self._get_stone_rigid_id()
        # Optimization variables for rigid body initial velocities
        initial_v = self.sim.rigid_v_arrays[0].numpy()[self.stone_rigid_id]
        initial_omega = self.sim.rigid_omega_arrays[0].numpy()[self.stone_rigid_id]
        device = self.sim.rigid_v_arrays[0].device
        self.opt_rigid_v = wp.array([initial_v], dtype=wp.vec3, device=device, requires_grad=True)
        self.opt_rigid_omega = wp.array([initial_omega], dtype=wp.vec3, device=device, requires_grad=True)
        
        self.target_x = None
        self.target_q = None
        
        self.init_targets()
        self.init_optimizer()

    def _get_stone_rigid_id(self):
        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []
        if len(rigid_bodies) > 0 and "objectId" in rigid_bodies[0]:
            return int(rigid_bodies[0]["objectId"])
        return 1

    def _get_stone_cfg(self):
        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []
        for rb in rigid_bodies:
            if int(rb.get("objectId", -1)) == self.stone_rigid_id:
                return rb
        return None

    def init_targets(self):
        if self.sim.num_objects > 0:
            self.target_rigid_x = wp.zeros_like(self.sim.rbs.rigid_x)
            self.target_rigid_q = wp.zeros_like(self.sim.rbs.rigid_quaternion)

            stone_cfg = self._get_stone_cfg()

            target_pos_np = self.sim.rbs.rigid_x.numpy().copy()
            if stone_cfg is not None and "targetX" in stone_cfg and self.stone_rigid_id < len(target_pos_np):
                target_pos_np[self.stone_rigid_id] = np.array(stone_cfg["targetX"], dtype=np.float32)

            wp.copy(self.target_rigid_x, wp.array(target_pos_np, dtype=wp.vec3))

            target_q_np = self.sim.rbs.rigid_quaternion.numpy().copy()
            if stone_cfg is not None and self.stone_rigid_id < len(target_q_np):
                if "targetQ" in stone_cfg:
                    target_q_np[self.stone_rigid_id] = np.array(stone_cfg["targetQ"], dtype=np.float32)

            wp.copy(self.target_rigid_q, wp.array(target_q_np, dtype=wp.quat))

    def init_optimizer(self):
        # 优化刚体1的初始速度和角速度（假设刚体1是石头）
        # 使用专门的优化变量
        self.opt_var = [self.opt_rigid_v, self.opt_rigid_omega]

        # Optimizer
        self.optimizer = wp.optim.Adam(self.opt_var, lr=self.sim.train_rate)
        
        # 将优化变量反向注册给 sim，以便SimSPH_diff能正确引用
        self.sim.opt_var = self.opt_var

    def compute_loss(self):
        # 使用 SimSPH_diff 中的数据和 kernel 计算 loss
        if self.sim.num_objects > 0:
            wp.launch(
                kernel=compute_rigid_loss,
                dim=self.sim.num_objects,
                inputs=[
                    self.sim.rigid_x_arrays[self.sim.sim_steps], 
                    self.target_rigid_x,
                    self.sim.rigid_v_arrays[self.sim.sim_steps], 
                    self.opt_rigid_v,
                    self.stone_rigid_id,
                    self.sim.rigid_quaternion_arrays[self.sim.sim_steps],
                    self.target_rigid_q,
                    self.sim.loss
                ]
            )

    def get_loss_state_info(self):
        info = {}
        if self.sim.num_objects > self.stone_rigid_id:
            target_idx = self.stone_rigid_id
            final_pos = self.sim.rigid_x_arrays[self.sim.sim_steps].numpy()[target_idx]
            target_pos = self.target_rigid_x.numpy()[target_idx]
            
            final_vel_y = self.sim.rigid_v_arrays[self.sim.sim_steps].numpy()[target_idx][1]
            
            info[f"stone_final_pos"] = final_pos
            info[f"stone_target_pos"] = target_pos
            info[f"stone_final_vy"] = final_vel_y
        return info

    def init_simulation_state(self):
        if self.sim.num_objects > 0: # TODO: 修改调用位置 
            # IMPORTANT: must stay in tape and avoid numpy() / re-allocations,
            # otherwise autograd graph from opt vars to simulation state is broken.
            with self.sim.tapes[-1]:
                wp.launch(
                    kernel=assign_rigid_init_state_kernel,
                    dim=1,
                    inputs=[
                        self.sim.rigid_v_arrays[0],
                        self.sim.rigid_omega_arrays[0],
                        self.opt_rigid_v,
                        self.opt_rigid_omega,
                        self.stone_rigid_id,
                    ],
                )

            # Update rbs pointers
            wp.copy(self.sim.rbs.rigid_v, self.sim.rigid_v_arrays[0])
            wp.copy(self.sim.rbs.rigid_omega, self.sim.rigid_omega_arrays[0])

    def clear_grad(self):
        if self.opt_rigid_v.grad:
            self.opt_rigid_v.grad.zero_()
        if self.opt_rigid_omega.grad:
            self.opt_rigid_omega.grad.zero_()

    def norm_final_grad(self, v_grad, materialMarks):
        # Normalize rigid body gradients if needed
        # For now, we can just normalize the opt_rigid_v and opt_rigid_omega gradients directly
        if self.opt_rigid_v.grad:
            grad_np = self.opt_rigid_v.grad.numpy()
            norm = np.linalg.norm(grad_np)
            if norm > 1e-10:
                self.opt_rigid_v.grad = wp.array(grad_np / norm, dtype=wp.vec3, device=self.opt_rigid_v.device)
                
        if self.opt_rigid_omega.grad:
            grad_np = self.opt_rigid_omega.grad.numpy()
            norm = np.linalg.norm(grad_np)
            if norm > 1e-10:
                self.opt_rigid_omega.grad = wp.array(grad_np / norm, dtype=wp.vec3, device=self.opt_rigid_omega.device)
