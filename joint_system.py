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
	inv_stiffness: wp.array(dtype=wp.float32)
	# hinge axis in world coordinates for joints that allow rotation only around this axis.
	# If zero vector, joint is fully constrained (no special hinge behavior).
	hinge_axis: wp.array(dtype=wp.vec3)


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

	# If hinge axis provided (non-zero), project angular corrections onto that axis
	axis = joints.hinge_axis[tid]
	axis_len2 = wp.dot(axis, axis)
	if axis_len2 > 1e-12:
		axis_n = axis * (1.0 / wp.sqrt(axis_len2))
		# retain only component along axis
		dp = wp.dot(delta_omega_parent, axis_n)
		delta_omega_parent = axis_n * dp
		dc = wp.dot(delta_omega_child, axis_n)
		delta_omega_child = axis_n * dc

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
