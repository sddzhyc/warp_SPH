# pyright: reportInvalidTypeForm=false
import warp as wp

from norm_grad_utils import norm_grad_quat, norm_grad_vec3
from rigid_fluid_coupling import RigidBodies


@wp.struct
class ArticulatedJoints:
	parent_index: wp.array(dtype=int)
	child_index: wp.array(dtype=int)
	parent_local: wp.array(dtype=wp.vec3)
	child_local: wp.array(dtype=wp.vec3)
	# Revolute Joint: hinge axes in parent/child local frames
	# (zero vector = spherical joint only, no orientation constraint)
	parent_axis_local: wp.array(dtype=wp.vec3)
	child_axis_local: wp.array(dtype=wp.vec3)
	# joint type: 0=spherical, 1=revolute (spherical + orientation)
	joint_type: wp.array(dtype=wp.int32)
	inv_stiffness: wp.array(dtype=wp.float32)


@wp.func
def quat_unit_conjugate(q: wp.quat):
	return wp.quat(-q[0], -q[1], -q[2], q[3])


@wp.func
def quat_relative(q: wp.quat, q0: wp.quat):
	return wp.normalize(q * quat_unit_conjugate(q0))


@wp.kernel
def integrate_rigid_velocity_inplace(
	bodies: RigidBodies,
	g: wp.vec3,
	dt: float,
):
	tid = wp.tid()
	f = bodies.rigid_force[tid] + g * bodies.rigid_mass[tid]
	bodies.rigid_v[tid] = bodies.rigid_v[tid] + dt * f * bodies.rigid_inv_mass[tid]
	bodies.rigid_omega[tid] = bodies.rigid_omega[tid] + dt * (bodies.rigid_inv_inertia[tid] @ bodies.rigid_torque[tid])


@wp.kernel
def integrate_rigid_pose_inplace(
	rigid_x: wp.array(dtype=wp.vec3),
	rigid_v: wp.array(dtype=wp.vec3),
	rigid_force: wp.array(dtype=wp.vec3),
	rigid_quaternion: wp.array(dtype=wp.quat),
	rigid_quaternion0: wp.array(dtype=wp.quat),
	rigid_omega: wp.array(dtype=wp.vec3),
	rigid_torque: wp.array(dtype=wp.vec3),
	rigid_inertia0: wp.array(dtype=wp.mat33),
	dt: float,
	rigid_x_out: wp.array(dtype=wp.vec3),
	rigid_v_out: wp.array(dtype=wp.vec3),
	rigid_force_out: wp.array(dtype=wp.vec3),
	rigid_quaternion_out: wp.array(dtype=wp.quat),
	rigid_omega_out: wp.array(dtype=wp.vec3),
	rigid_torque_out: wp.array(dtype=wp.vec3),
	rigid_inertia_out: wp.array(dtype=wp.mat33),
	rigid_inv_inertia_out: wp.array(dtype=wp.mat33),
):
	tid = wp.tid()
	v = rigid_v[tid]
	omega = rigid_omega[tid]

	rigid_x_out[tid] = rigid_x[tid] + dt * v
	rigid_v_out[tid] = v
	rigid_omega_out[tid] = omega

	q = rigid_quaternion[tid]
	dq = 0.5 * wp.quat(omega[0], omega[1], omega[2], 0.0) * q
	q = wp.normalize(q + dt * dq)
	rigid_quaternion_out[tid] = q

	q_rel = quat_relative(q, rigid_quaternion0[tid])
	R = wp.quat_to_matrix(q_rel)
	I = R @ rigid_inertia0[tid] @ wp.transpose(R)
	rigid_inertia_out[tid] = I
	det_I = wp.determinant(I)
	if wp.abs(det_I) > 1.0e-8:
		rigid_inv_inertia_out[tid] = wp.inverse(I)
	else:
		rigid_inv_inertia_out[tid] = wp.mat33(
			0.0, 0.0, 0.0,
			0.0, 0.0, 0.0,
			0.0, 0.0, 0.0,
		)

	rigid_force_out[tid] = wp.vec3(0.0, 0.0, 0.0)
	rigid_torque_out[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def build_tangent_basis(n: wp.vec3):
	"""给定单位向量 n，构造两个正交切向量 t1 ⟂ n, t2 ⟂ n（t1 ⟂ t2）
	参考论文 4.3 节的 Orientation Joint 投影矩阵构造"""
	# 选一个与 n 不共线的参考向量来叉积
	if wp.abs(n[0]) < wp.float32(0.999):
		ref = wp.vec3(1.0, 0.0, 0.0)
	else:
		ref = wp.vec3(0.0, 1.0, 0.0)
	t1 = wp.normalize(wp.cross(ref, n))
	t2 = wp.cross(n, t1)
	return t1, t2


@wp.func
def solve_orientation_joint(
	q_parent_rel: wp.quat,
	q_child_rel: wp.quat,
	invI_parent: wp.mat33,
	invI_child: wp.mat33,
	axis_parent_local: wp.vec3,
	axis_child_local: wp.vec3,
	dt: float,
):
	"""论文 4.3 节 Orientation Joint 求解器 (Revolute Joint 的角度约束部分)

	约束: parent 和 child 的局部轴在旋转后应保持同向（世界空间重合）
	仅允许绕 a1 的旋转，在 t1/t2 切空间方向施加约束

	Returns:
		(delta_omega_parent, delta_omega_child): 角速度修正量
	"""
	# 1. 计算世界坐标轴
	a1 = wp.quat_to_matrix(q_parent_rel) @ axis_parent_local
	a2 = wp.quat_to_matrix(q_child_rel) @ axis_child_local
	a1 = wp.normalize(a1)
	a2 = wp.normalize(a2)

	# 2. 论文误差: d = a1 × a2
	rot_error = wp.cross(a1, a2)

	# 3. 构造 a1 的切空间基 (t1, t2 ⟂ a1)
	t1, t2 = build_tangent_basis(a1)

	# 4. 将误差投影到 2D 约束空间: rhs = P · rot_error / dt
	#    P = [t1^T; t2^T] (2×3)，不显式构造矩阵，直接点积
	rhs0 = wp.dot(rot_error, t1) / dt
	rhs1 = wp.dot(rot_error, t2) / dt

	# 5. L = invI_parent + invI_child (3×3)
	L = invI_parent + invI_child

	# 6. 投影后的 2×2 有效质量矩阵 M = P · L · P^T
	m00 = wp.dot(t1, L @ t1)
	m01 = wp.dot(t1, L @ t2)
	m10 = m01
	m11 = wp.dot(t2, L @ t2)

	# 7. 求解 M · λ = rhs (2×2 矩阵求逆)
	detM = m00 * m11 - m01 * m10
	eps = wp.float32(1e-12)
	# 用条件赋值避免 NaN（若奇异则返回零修正）
	if wp.abs(detM) < eps:
		z = wp.vec3(0.0, 0.0, 0.0)
		return z, z

	invM00 = m11 / detM
	invM01 = -m01 / detM
	invM10 = -m10 / detM
	invM11 = m00 / detM

	lambda0 = invM00 * rhs0 + invM01 * rhs1
	lambda1 = invM10 * rhs0 + invM11 * rhs1

	# 8. 恢复 3D 角冲量: l = P^T · λ = λ0·t1 + λ1·t2
	ang_impulse = lambda0 * t1 + lambda1 * t2

	# 9. 返回角速度修正
	delta_parent = invI_parent @ ang_impulse
	delta_child = -invI_child @ ang_impulse

	return delta_parent, delta_child


@wp.kernel
def solve_articulated_joints_impulse_inplace(
	rigid_x: wp.array(dtype=wp.vec3),
	rigid_quaternion: wp.array(dtype=wp.quat),
	rigid_quaternion0: wp.array(dtype=wp.quat),
	rigid_inv_mass: wp.array(dtype=wp.float32),
	rigid_inv_inertia: wp.array(dtype=wp.mat33),
	joints: ArticulatedJoints,
	num_bodies: int,
	num_joints: int,
	dt: float,
	rigid_delta_v_out: wp.array(dtype=wp.vec3),
	rigid_delta_omega_out: wp.array(dtype=wp.vec3),
):
	tid = wp.tid()
	if tid >= num_joints:
		return

	I3 = wp.diag(wp.vec3(1.0, 1.0, 1.0))
	parent = joints.parent_index[tid]
	child = joints.child_index[tid]

	if parent < 0 or parent >= num_bodies or child < 0 or child >= num_bodies:
		return

	q_parent_rel = quat_relative(rigid_quaternion[parent], rigid_quaternion0[parent])
	q_child_rel = quat_relative(rigid_quaternion[child], rigid_quaternion0[child])
	RL1 = wp.quat_to_matrix(q_parent_rel) @ joints.parent_local[tid]
	RL2 = wp.quat_to_matrix(q_child_rel) @ joints.child_local[tid]

	delta_x = (rigid_x[parent] + RL1) - (rigid_x[child] + RL2)

	K_parent = rigid_inv_mass[parent] * I3 - wp.skew(RL1) @ rigid_inv_inertia[parent] @ wp.skew(RL1)
	K_child = rigid_inv_mass[child] * I3 - wp.skew(RL2) @ rigid_inv_inertia[child] @ wp.skew(RL2)
	K = K_parent + K_child
	det_K = wp.determinant(K)
	if wp.abs(det_K) <= 1.0e-8:
		return
	invK = wp.inverse(K)

	impulse = (invK @ delta_x) / dt
	delta_v_parent = -impulse * rigid_inv_mass[parent]
	delta_v_child = impulse * rigid_inv_mass[child]
	delta_omega_parent = rigid_inv_inertia[parent] @ wp.cross(RL1, -impulse)
	delta_omega_child = rigid_inv_inertia[child] @ wp.cross(RL2, impulse)

	# --- Orientation Joint (Revolute) 角速度约束 ---
	# joint_type == 1 (revolute): 在球关节基础上叠加 orientation joint
	# 约束 parent 和 child 的局部轴保持同向，仅允许绕该轴的 1-DOF 旋转
	if joints.joint_type[tid] == 1:
		rot_domega_parent, rot_domega_child = solve_orientation_joint(
			q_parent_rel,
			q_child_rel,
			rigid_inv_inertia[parent],
			rigid_inv_inertia[child],
			joints.parent_axis_local[tid],
			joints.child_axis_local[tid],
			dt,
		)
		delta_omega_parent = delta_omega_parent + rot_domega_parent
		delta_omega_child = delta_omega_child + rot_domega_child

	wp.atomic_add(rigid_delta_v_out, parent, delta_v_parent)
	wp.atomic_add(rigid_delta_v_out, child, delta_v_child)

	wp.atomic_add(rigid_delta_omega_out, parent, delta_omega_parent)
	wp.atomic_add(rigid_delta_omega_out, child, delta_omega_child)


@wp.kernel
def compose_joint_velocity_from_delta(
	rigid_v_for_joint: wp.array(dtype=wp.vec3),
	rigid_omega_for_joint: wp.array(dtype=wp.vec3),
	rigid_delta_v: wp.array(dtype=wp.vec3),
	rigid_delta_omega: wp.array(dtype=wp.vec3),
	rigid_v_out: wp.array(dtype=wp.vec3),
	rigid_omega_out: wp.array(dtype=wp.vec3),
):
	tid = wp.tid()
	rigid_v_out[tid] = rigid_v_for_joint[tid] + rigid_delta_v[tid]
	rigid_omega_out[tid] = rigid_omega_for_joint[tid] + rigid_delta_omega[tid]


@wp.kernel
def integrate_rigid_velocity_diff(
	rigid_v: wp.array(dtype=wp.vec3),
	rigid_omega: wp.array(dtype=wp.vec3),
	rigid_force: wp.array(dtype=wp.vec3),
	rigid_torque: wp.array(dtype=wp.vec3),
	rigid_mass: wp.array(dtype=wp.float32),
	rigid_inv_mass: wp.array(dtype=wp.float32),
	rigid_inv_inertia: wp.array(dtype=wp.mat33),
	g: wp.vec3,
	dt: float,
	rigid_v_out: wp.array(dtype=wp.vec3),
	rigid_omega_out: wp.array(dtype=wp.vec3),
):
	tid = wp.tid()
	f = rigid_force[tid] + g * rigid_mass[tid]

	rigid_v_out[tid] = norm_grad_vec3(rigid_v[tid] + dt * f * rigid_inv_mass[tid])
	rigid_omega_out[tid] = norm_grad_vec3(rigid_omega[tid] + dt * (rigid_inv_inertia[tid] @ rigid_torque[tid]))


@wp.kernel
def integrate_rigid_pose_diff_inplace(
	rigid_x: wp.array(dtype=wp.vec3),
	rigid_v: wp.array(dtype=wp.vec3),
	rigid_quaternion: wp.array(dtype=wp.quat),
	rigid_quaternion0: wp.array(dtype=wp.quat),
	rigid_omega: wp.array(dtype=wp.vec3),
	rigid_inertia0: wp.array(dtype=wp.mat33),
	dt: float,
	rigid_x_out: wp.array(dtype=wp.vec3),
	rigid_force_out: wp.array(dtype=wp.vec3),
	rigid_quaternion_out: wp.array(dtype=wp.quat),
	rigid_torque_out: wp.array(dtype=wp.vec3),
	rigid_inertia_out: wp.array(dtype=wp.mat33),
	rigid_inv_inertia_out: wp.array(dtype=wp.mat33),
):
	tid = wp.tid()

	rigid_x_out[tid] = norm_grad_vec3(rigid_x[tid] + dt * rigid_v[tid])
	q = rigid_quaternion[tid]
	omega = rigid_omega[tid]
	dq = 0.5 * wp.quat(omega[0], omega[1], omega[2], 0.0) * q
	q_normalized = wp.normalize(q + dt * dq)
	rigid_quaternion_out[tid] = norm_grad_quat(q_normalized)

	q_rel = quat_relative(q_normalized, rigid_quaternion0[tid])
	R = wp.quat_to_matrix(q_rel)
	I = R @ rigid_inertia0[tid] @ wp.transpose(R)
	rigid_inertia_out[tid] = I
	rigid_inv_inertia_out[tid] = wp.inverse(I)

	rigid_force_out[tid] = wp.vec3(0.0, 0.0, 0.0)
	rigid_torque_out[tid] = wp.vec3(0.0, 0.0, 0.0)
