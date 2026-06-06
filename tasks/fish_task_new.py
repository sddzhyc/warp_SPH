# pyright: reportInvalidTypeForm=false
import numpy as np
import warp as wp

from .fish_task import FishTask


@wp.kernel
def compute_fish_body_tail_loss_kernel(
    rigid_q: wp.array(dtype=wp.quat),
    target_rigid_q: wp.array(dtype=wp.quat),
    body_rigid_id: int,
    tail_rigid_id: int,
    loss: wp.array(dtype=float),
):
    """计算鱼身 + 鱼尾的四元数姿态 loss 之和"""
    tid = wp.tid()
    if tid == 0:
        # body loss
        qb = rigid_q[body_rigid_id]
        tqb = target_rigid_q[body_rigid_id]
        dqb = qb - tqb
        lb = 0.5 * wp.dot(dqb, dqb)

        # tail loss
        qt = rigid_q[tail_rigid_id]
        tqt = target_rigid_q[tail_rigid_id]
        dqt = qt - tqt
        lt = 0.5 * wp.dot(dqt, dqt)

        wp.atomic_add(loss, 0, lb + lt)


@wp.kernel
def compute_fish_forward_velocity_loss_kernel(
    rigid_q: wp.array(dtype=wp.quat),
    rigid_v: wp.array(dtype=wp.vec3),
    body_rigid_id: int,
    forward_velocity_weight: float,
    loss: wp.array(dtype=float),
):
    """计算鱼身线速度沿朝向（局部 X 轴）分量的 loss，激励鱼向前游动"""
    tid = wp.tid()
    if tid == 0:
        q = rigid_q[body_rigid_id]
        v = rigid_v[body_rigid_id]
        # 局部 X 轴（鱼头方向）变换到世界空间
        forward = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
        forward_vel = wp.dot(forward, v)
        # 最大化前向速度 = 最小化负前向速度
        wp.atomic_add(loss, 0, -forward_velocity_weight * forward_vel)


class FishTaskNew(FishTask):
    """专门用于 fish_on_water_new 场景（body + tail 两段关节鱼）

    与 FishTask 的区别:
    - loss = body 姿态 loss + tail 姿态 loss + 前向线速度 loss
    - 由 FishEnv 的 action 模式自动切换为关节力矩（施加在 body 和 tail 上大小相等方向相反的 Y 轴角速度）
    """

    def __init__(self, sim, use_v_loss=False):
        super().__init__(sim)
        self.use_v_loss = use_v_loss
        self.forward_velocity_weight = 1.0  # 前向线速度 loss 权重

    def init_targets(self):
        """为 body 和 tail 都设置目标四元数"""
        self.target_rigid_q = wp.zeros_like(self.sim.rbs.rigid_quaternion)
        target_q_np = self.sim.rbs.rigid_quaternion.numpy().copy()

        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []

        import scipy.spatial.transform as st

        for i, rb in enumerate(rigid_bodies):
            obj_id = int(rb.get("objectId", i + 1))
            if obj_id not in (self.body_rigid_id, self.tail_rigid_id):
                continue

            # 从 JSON 读取目标四元数（优先 targetQuaternion，fallback targetAngleInDegree）
            if "targetQuaternion" in rb:
                q = np.array(rb["targetQuaternion"], dtype=np.float32)
            elif "targetQ" in rb:
                q = np.array(rb["targetQ"], dtype=np.float32)
            elif "targetAngleInDegree" in rb:
                euler = rb["targetAngleInDegree"]
                q = st.Rotation.from_euler('xyz', euler, degrees=True).as_quat().astype(np.float32)
            else:
                continue
            target_q_np[obj_id] = q

        wp.copy(self.target_rigid_q, wp.array(target_q_np, dtype=wp.quat))

    def compute_loss(self, end_step=None):
        """body + tail 四元数姿态 loss + 前向线速度 loss"""
        if self.sim.num_objects <= 0:
            return
        if end_step is None:
            end_step = self.sim.sim_steps
        wp.launch(
            kernel=compute_fish_body_tail_loss_kernel,
            dim=1,
            inputs=[
                self.sim.rigid_quaternion_arrays[end_step],
                self.target_rigid_q,
                self.body_rigid_id,
                self.tail_rigid_id,
            ],
            outputs=[self.sim.loss],
        )

        # 前向线速度 loss：激励鱼沿朝向（局部 X 轴 → 世界 X 轴方向）向前游动
        if self.use_v_loss:
            wp.launch(
                kernel=compute_fish_forward_velocity_loss_kernel,
                dim=1,
                inputs=[
                    self.sim.rigid_quaternion_arrays[end_step],
                    self.sim.rigid_v_arrays[end_step],
                    self.body_rigid_id,
                    self.forward_velocity_weight,
                ],
                outputs=[self.sim.loss],
            )
