import warp as wp
import numpy as np
import trimesh as tm
from functools import reduce
from config_builder import SimConfig

class CPUPrefixSumExecutor:
    def __init__(self, length) -> None:
        self._length = length - 1
    def run(self, input_arr):
        # if input_arr.dtype != ti.i32:
        #     raise RuntimeError("Only ti.i32 type is supported for prefix sum.")
        for i in range(self._length):
            input_arr[i + 1] += input_arr[i]

# @ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domian_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)
        # currently only 3-dim simulations are supported
        assert self.dim == 3
        # Simulation method
        self.simulation_method = self.cfg.get_cfg("simulationMethod")

        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim
        # particle count (scalar on host)
        self.particle_num = 0

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()

        #========== Compute number of particles ==========#
        #### Process Fluid Blocks ####
        fluid_blocks = self.cfg.get_fluid_blocks()
        fluid_particle_num = 0
        for fluid in fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num

        #### Process Rigid Blocks ####
        rigid_blocks = self.cfg.get_rigid_blocks()
        rigid_particle_num = 0
        for rigid in rigid_blocks:
            particle_num = self.compute_cube_particle_num(rigid["start"], rigid["end"])
            rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num
        
        #### Process Rigid Bodies ####
        rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in rigid_bodies:
            voxelized_points_np = self.load_rigid_body(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_particle_num += voxelized_points_np.shape[0]
        
        self.fluid_particle_num = fluid_particle_num
        self.solid_particle_num = rigid_particle_num
        self.particle_max_num = fluid_particle_num + rigid_particle_num
        self.num_rigid_bodies = len(rigid_blocks)+len(rigid_bodies)

        self.num_objects = self.num_rigid_bodies + len(fluid_blocks)
        if len(rigid_blocks) > 0:
            print("Warning: currently rigid block functions are not completed, may lead to unexpected behaviour")
            input("Press Enter to continue")

        #### TODO: Handle the Particle Emitter ####
        # self.particle_max_num += emitted particles
        print(f"Current particle num: {self.particle_num}, Particle max num: {self.particle_max_num}")

        #========== Allocate memory ==========#
        # Rigid body properties
        if self.num_rigid_bodies > 0:
            # TODO: Here we actually only need to store rigid boides, however the object id of rigid may not start from 0, so allocate center of mass for all objects
            # Warp arrays for rigid body properties
            # Use vector/matrix dtypes where appropriate. Warp defaults to float32.
            # Each is sized by number of objects (indexable by object id).
            self.rigid_rest_cm = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.rigid_x = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.rigid_v0 = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.rigid_v = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.rigid_quaternion = np.zeros((self.num_objects, 4), dtype=np.float32)
            self.rigid_omega = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.rigid_omega0 = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.rigid_force = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.rigid_torque = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.rigid_mass = np.zeros(self.num_objects, dtype=float)
            # use explicit 3x3 matrices as numpy arrays (num_objects x dim x dim)
            self.rigid_inertia = np.zeros((self.num_objects, self.dim, self.dim), dtype=np.float32)
            self.rigid_inertia0 = np.zeros((self.num_objects, self.dim, self.dim), dtype=np.float32)
            self.rigid_inv_mass = np.zeros(self.num_objects, dtype=float)
            self.rigid_inv_inertia = np.zeros((self.num_objects, self.dim, self.dim), dtype=np.float32)

        # Particle num of each grid (use numpy arrays for prefix-sum ops)
        self.grid_count = int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2])
        self.grid_particles_num = np.zeros(self.grid_count, dtype=np.int32)
        self.grid_particles_num_temp = np.zeros(self.grid_count, dtype=np.int32)
        # TODO:改用warp内核版本的prefix_sum
        self.prefix_sum_executor = CPUPrefixSumExecutor(self.grid_particles_num.shape[0])
        # Particle related properties (use Warp arrays for vectors/scalars)
        self.object_id = np.zeros(self.particle_max_num, dtype=int)
        self.x = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.x_0 = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.v = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.acceleration = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.m_V = np.zeros(self.particle_max_num, dtype=float)
        self.m = np.zeros(self.particle_max_num, dtype=float)
        self.density = np.zeros(self.particle_max_num, dtype=float)
        self.pressure = np.zeros(self.particle_max_num, dtype=float)
        self.material = np.zeros(self.particle_max_num, dtype=int)
        self.color = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.is_dynamic = np.zeros(self.particle_max_num, dtype=int)

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor = np.zeros(self.particle_max_num, dtype=float)
            self.density_adv = np.zeros(self.particle_max_num, dtype=float)

        # Buffers for sort (warp arrays)
        self.object_id_buffer = np.zeros(self.particle_max_num, dtype=int)
        self.x_buffer = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.x_0_buffer = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.v_buffer = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.acceleration_buffer = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.m_V_buffer = np.zeros(self.particle_max_num, dtype=float)
        self.m_buffer = np.zeros(self.particle_max_num, dtype=float)
        self.density_buffer = np.zeros(self.particle_max_num, dtype=float)
        self.pressure_buffer = np.zeros(self.particle_max_num, dtype=float)
        self.material_buffer = np.zeros(self.particle_max_num, dtype=int)
        self.color_buffer = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        self.is_dynamic_buffer = np.zeros(self.particle_max_num, dtype=int)

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor_buffer = np.zeros(self.particle_max_num, dtype=float)
            self.density_adv_buffer = np.zeros(self.particle_max_num, dtype=float)

        # Grid id for each particle (warp arrays)
        self.grid_ids = np.zeros(self.particle_max_num, dtype=int)
        self.grid_ids_buffer = np.zeros(self.particle_max_num, dtype=int)
        self.grid_ids_new = np.zeros(self.particle_max_num, dtype=int)

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
            self.color_vis_buffer = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)


        #========== Initialize particles ==========#

        # Fluid block
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=1, # enforce fluid dynamic
                          color=color,
                          material=1) # 1 indicates fluid
        
        # TODO: Handle rigid block
        # Rigid block
        for rigid in rigid_blocks:
            obj_id = rigid["objectId"]
            offset = np.array(rigid["translation"])
            start = np.array(rigid["start"]) + offset
            end = np.array(rigid["end"]) + offset
            scale = np.array(rigid["scale"])
            velocity = rigid["velocity"]
            angular_velocity = rigid["angularVelocity"]
            density = rigid["density"]
            color = rigid["color"]
            is_dynamic = rigid["isDynamic"]
            self.rigid_v[obj_id] = velocity
            self.rigid_v0[obj_id] = velocity
            self.rigid_omega[obj_id] = angular_velocity
            self.rigid_omega0[obj_id] = angular_velocity
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=is_dynamic,
                          color=color,
                          material=0) # 1 indicates solid

        # Rigid bodies
        for rigid_body in rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]
            if is_dynamic:
                velocity = np.array(rigid_body["velocity"], dtype=np.float32)
                if "angularVelocity" in rigid_body:
                    angular_velocity = np.array(rigid_body["angularVelocity"], dtype=np.float32)
                else:
                    angular_velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
                angular_velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            density = rigid_body["density"]
            color = np.array(rigid_body["color"], dtype=np.int32)
            self.rigid_v[obj_id] = velocity
            self.rigid_v0[obj_id] = velocity
            self.rigid_omega[obj_id] = angular_velocity
            self.rigid_omega0[obj_id] = angular_velocity
            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32), # position
                               np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32), # density
                               np.zeros(num_particles_obj, dtype=np.float32), # pressure
                               np.array([0 for _ in range(num_particles_obj)], dtype=np.int32), # material is solid
                               is_dynamic * np.ones(num_particles_obj, dtype=np.int32), # is_dynamic
                               np.stack([color for _ in range(num_particles_obj)])) # color


    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        # host-side helper to set particle data
        # Debug: print type of self.object_id to help diagnose assignment errors
        try:
            t = type(self.object_id)
        except Exception:
            t = None
        print(f"[debug] add_particle: self.object_id type = {t}")
        self.object_id[p] = int(obj_id)
        # store vectors as tuples (warp will accept them)
        self.x[p] = tuple(float(xi) for xi in x)
        self.x_0[p] = tuple(float(xi) for xi in x)
        self.v[p] = tuple(float(vi) for vi in v)
        self.density[p] = float(density)
        self.m_V[p] = float(self.m_V0)
        self.m[p] = float(self.m_V0 * density)
        self.pressure[p] = float(pressure)
        self.material[p] = int(material)
        self.is_dynamic[p] = int(is_dynamic)
        self.color[p] = tuple(int(c) for c in color)
    
    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_is_dynamic,
                      new_particles_color
                      ):
        # delegate to host-side implementation
        self._add_particles(object_id,
                            new_particles_num,
                            new_particles_positions,
                            new_particles_velocity,
                            new_particle_density,
                            new_particle_pressure,
                            new_particles_material,
                            new_particles_is_dynamic,
                            new_particles_color
                            )
    def _add_particles(self,
                       object_id: int,
                       new_particles_num: int,
                       new_particles_positions,
                       new_particles_velocity,
                       new_particle_density,
                       new_particle_pressure,
                       new_particles_material,
                       new_particles_is_dynamic,
                       new_particles_color):
        # host-side particle addition (works with numpy arrays or array-like inputs)
        start = self.particle_num
        for i in range(new_particles_num):
            p = start + i
            x = new_particles_positions[i]
            v = new_particles_velocity[i]
            density = new_particle_density[i]
            pressure = new_particle_pressure[i]
            material = int(new_particles_material[i])
            is_dyn = int(new_particles_is_dynamic[i])
            color = new_particles_color[i]
            self.add_particle(p, object_id, x, v, density, pressure, material, is_dyn, color)
        self.particle_num += new_particles_num


    def pos_to_index(self, pos):
        # pos may be a warp vec or sequence; convert to numpy then to int indices
        pos_np = np.array([pos[0], pos[1], pos[2]], dtype=np.float64)
        return (pos_np / self.grid_size).astype(int)


    def flatten_grid_index(self, grid_index):
        return int(grid_index[0]) * int(self.grid_num[1]) * int(self.grid_num[2]) + int(grid_index[1]) * int(self.grid_num[2]) + int(grid_index[2])

    def get_flatten_grid_index(self, pos):
        idx = self.pos_to_index(pos)
        return self.flatten_grid_index(idx)
    

    def is_static_rigid_body(self, p):
        return int(self.material[p]) == self.material_solid and (not int(self.is_dynamic[p]))


    def is_dynamic_rigid_body(self, p):
        return int(self.material[p]) == self.material_solid and int(self.is_dynamic[p])
    

    def update_grid_id(self):
        # host implementation: recompute grid ids and counts
        self.grid_particles_num.fill(0)
        for i in range(self.particle_num):
            pos = self.x[i]
            grid_index = self.get_flatten_grid_index(pos)
            self.grid_ids[i] = int(grid_index)
            self.grid_particles_num[grid_index] += 1
        # copy temp
        self.grid_particles_num_temp[:] = self.grid_particles_num[:]
    
    def counting_sort(self):
        # Host-side counting sort based on grid_ids and grid_particles_num (which should be prefix-summed)
        # Note: grid_particles_num is inclusive prefix sums after calling parallel_prefix_sum_inclusive_inplace
        # Compute new positions
        # make a temp array for decrementing counts
        temp = self.grid_particles_num_temp.copy()
        # iterate particles in reverse order
        for I in range(self.particle_max_num - 1, -1, -1):
            if I >= self.particle_num:
                continue
            gid = int(self.grid_ids[I])
            base_offset = 0
            if gid - 1 >= 0:
                base_offset = int(self.grid_particles_num[gid - 1])
            temp[gid] -= 1
            new_index = int(temp[gid]) + base_offset
            self.grid_ids_new[I] = new_index

        # scatter into buffers
        # FIXME: make it the actual particle num
        for I in range(self.particle_max_num):
            new_index = int(self.grid_ids_new[I])
            self.grid_ids_buffer[new_index] = int(self.grid_ids[I])
            self.object_id_buffer[new_index] = int(self.object_id[I])
            self.x_0_buffer[new_index] = tuple(self.x_0[I])
            self.x_buffer[new_index] = tuple(self.x[I])
            self.v_buffer[new_index] = tuple(self.v[I])
            self.acceleration_buffer[new_index] = tuple(self.acceleration[I])
            self.m_V_buffer[new_index] = float(self.m_V[I])
            self.m_buffer[new_index] = float(self.m[I])
            self.density_buffer[new_index] = float(self.density[I])
            self.pressure_buffer[new_index] = float(self.pressure[I])
            self.material_buffer[new_index] = int(self.material[I])
            self.color_buffer[new_index] = tuple(self.color[I])
            self.is_dynamic_buffer[new_index] = int(self.is_dynamic[I])
            if self.simulation_method == 4:
                self.dfsph_factor_buffer[new_index] = float(self.dfsph_factor[I])
                self.density_adv_buffer[new_index] = float(self.density_adv[I])

        # copy back from buffers
        for I in range(self.particle_num):
            self.grid_ids[I] = int(self.grid_ids_buffer[I])
            self.object_id[I] = int(self.object_id_buffer[I])
            self.x_0[I] = tuple(self.x_0_buffer[I])
            self.x[I] = tuple(self.x_buffer[I])
            self.v[I] = tuple(self.v_buffer[I])
            self.acceleration[I] = tuple(self.acceleration_buffer[I])
            self.m_V[I] = float(self.m_V_buffer[I])
            self.m[I] = float(self.m_buffer[I])
            self.density[I] = float(self.density_buffer[I])
            self.pressure[I] = float(self.pressure_buffer[I])
            self.material[I] = int(self.material_buffer[I])
            self.color[I] = tuple(self.color_buffer[I])
            self.is_dynamic[I] = int(self.is_dynamic_buffer[I])
            if self.simulation_method == 4:
                self.dfsph_factor[I] = float(self.dfsph_factor_buffer[I])
                self.density_adv[I] = float(self.density_adv_buffer[I])
    

    def initialize_particle_system(self):
        # compute grid ids and counts
        self.update_grid_id()
        # run inclusive prefix sum on grid_particles_num -> produces inclusive prefix sums
        self.prefix_sum_executor.run(self.grid_particles_num)
        # copy into temp for counting sort
        self.grid_particles_num_temp[:] = self.grid_particles_num[:]
        # self.counting_sort()
    

    def for_all_neighbors(self, p_i, task, ret=None):
        # iterate neighbor cells on host and call task(p_i, p_j, ret)
        center_cell = self.pos_to_index(self.x[p_i])
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    grid_index = self.flatten_grid_index(center_cell + np.array([ox, oy, oz]))
                    start = 0 if grid_index - 1 < 0 else int(self.grid_particles_num[grid_index - 1])
                    end = int(self.grid_particles_num[grid_index])
                    for p_j in range(start, end):
                        if p_i != p_j:
                            xi = np.array([self.x[p_i][0], self.x[p_i][1], self.x[p_i][2]])
                            xj = np.array([self.x[p_j][0], self.x[p_j][1], self.x[p_j][2]])
                            if np.linalg.norm(xi - xj) < self.support_radius:
                                task(p_i, p_j, ret)

    def copy_to_numpy(self, np_arr, src_arr):
        for i in range(self.particle_num):
            np_arr[i] = src_arr[i]
    
    def copy_to_vis_buffer(self, invisible_objects=[]):
        if len(invisible_objects) != 0:
            self.x_vis_buffer.fill(0.0)
            self.color_vis_buffer.fill(0.0)
        for obj_id in self.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)

    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        # host-side copy for visualization buffers
        for i in range(self.particle_max_num):
            if int(self.object_id[i]) == obj_id:
                self.x_vis_buffer[i] = tuple(self.x[i])
                col = np.array(self.color[i], dtype=np.float32) / 255.0
                self.color_vis_buffer[i] = tuple(col)

    def dump(self, obj_id):
        np_object_id = self.object_id
        mask = (np_object_id == obj_id).nonzero()
        np_x = self.x[mask]
        np_v = self.v_buffer[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }
    

    def load_rigid_body(self, rigid_body):
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])
        offset = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = rigid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
        
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        rigid_body["mesh"] = mesh_backup
        rigid_body["restPosition"] = mesh_backup.vertices
        rigid_body["restCenterOfMass"] = mesh_backup.vertices.mean(axis=0)
        is_success = tm.repair.fill_holes(mesh)
            # print("Is the mesh successfully repaired? ", is_success)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        # voxelized_mesh.show()
        voxelized_points_np = voxelized_mesh.points
        print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")
        
        return voxelized_points_np


    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i], self.particle_diameter))
        return reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])

    def add_cube(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 material,
                 is_dynamic,
                 color=(0,0,0),
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        print('particle num ', num_new_particles)

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)


    # add for debug
    def print_rigid_info(self):
        for r in self.object_id_rigid_body:
            print("object ", r)
            print("x", self.rigid_x[r])
            print("x0", self.rigid_rest_cm[r])
            print("v", self.rigid_v[r])
            print("v0", self.rigid_v0[r])
            print("w", self.rigid_omega[r])
            print("w0", self.rigid_omega0[r])
            print("q", self.rigid_quaternion[r])

            print("m", self.rigid_mass[r])
            print("I", self.rigid_inertia[r])
            print("f", self.rigid_force[r])
            print("t", self.rigid_torque[r])

    def initialize_rigid_info(self):
        # self.print_rigid_info()
        # call in initialization after compute_rigid_rest_cm
        # for r_obj_id in len(self.rigid_x):
        # no rigid processs
        for r_obj_id in range(len(self.rigid_x)):
            # velocities and angular velocities have already been initialized in particle system
            self.rigid_x[r_obj_id] = self.rigid_rest_cm[r_obj_id]
            self.rigid_quaternion[r_obj_id] = np.array([1.0, 0.0, 0.0, 0.0])
            self.rigid_force[r_obj_id].fill(0.0)
            self.rigid_torque[r_obj_id].fill(0.0)