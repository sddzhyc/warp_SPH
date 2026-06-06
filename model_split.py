
import numpy as np

def read_obj_file(filename):
    vertices = []
    texture_coords = []
    normals = []
    faces = []
    material_lib = None
    material = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt':
                texture_coords.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'vn':
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                faces.append(parts[1:])
            elif parts[0] == 'mtllib':
                material_lib = parts[1]
            elif parts[0] == 'usemtl':
                material = parts[1]
    
    return vertices, texture_coords, normals, faces, material_lib, material

def split_model(vertices, texture_coords, normals, faces):
    vertices_array = np.array(vertices)
    
    x_coords = vertices_array[:, 0]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    split_x = (min_x + max_x) / 2
    
    body_vertex_indices = []
    tail_vertex_indices = []
    
    for i, x in enumerate(x_coords):
        if x >= split_x:
            body_vertex_indices.append(i + 1)
        else:
            tail_vertex_indices.append(i + 1)
    
    body_vertex_set = set(body_vertex_indices)
    tail_vertex_set = set(tail_vertex_indices)
    
    body_faces = []
    tail_faces = []
    
    for face in faces:
        vertex_indices = []
        for vertex_data in face:
            parts = vertex_data.split('/')
            vertex_indices.append(int(parts[0]))
        
        body_count = sum(1 for v in vertex_indices if v in body_vertex_set)
        tail_count = sum(1 for v in vertex_indices if v in tail_vertex_set)
        
        if body_count >= tail_count:
            body_faces.append(face)
        else:
            tail_faces.append(face)
    
    return body_vertex_indices, tail_vertex_indices, body_faces, tail_faces

def reindex_and_write_obj(filename, vertex_indices, faces, all_vertices, all_texture_coords, all_normals, material_lib, material):
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices, 1)}
    
    used_vertices = [all_vertices[old_idx - 1] for old_idx in vertex_indices]
    
    used_texture_indices = set()
    used_normal_indices = set()
    
    processed_faces = []
    for face in faces:
        new_face = []
        for vertex_data in face:
            parts = vertex_data.split('/')
            old_vertex_idx = int(parts[0])
            
            if old_vertex_idx in vertex_map:
                new_vertex_idx = vertex_map[old_vertex_idx]
                
                texture_part = parts[1] if len(parts) > 1 and parts[1] else ''
                normal_part = parts[2] if len(parts) > 2 and parts[2] else ''
                
                if texture_part:
                    used_texture_indices.add(int(texture_part))
                if normal_part:
                    used_normal_indices.add(int(normal_part))
                
                new_face.append(f"{new_vertex_idx}/{texture_part}/{normal_part}")
        
        if new_face:
            processed_faces.append(new_face)
    
    texture_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_texture_indices), 1)}
    normal_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_normal_indices), 1)}
    
    used_texture_coords = [all_texture_coords[old_idx - 1] for old_idx in sorted(used_texture_indices)]
    used_normals = [all_normals[old_idx - 1] for old_idx in sorted(used_normal_indices)]
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Split OBJ file\n")
        if material_lib:
            f.write(f"mtllib {material_lib}\n")
        f.write("o split_object\n")
        
        for v in used_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        for vt in used_texture_coords:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        
        for vn in used_normals:
            f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
        
        if material:
            f.write(f"usemtl {material}\n")
        f.write("s 1\n")
        
        for face in processed_faces:
            new_face_str = []
            for vertex_data in face:
                parts = vertex_data.split('/')
                new_vertex_idx = parts[0]
                
                texture_part = parts[1] if len(parts) > 1 and parts[1] else ''
                normal_part = parts[2] if len(parts) > 2 and parts[2] else ''
                
                if texture_part:
                    new_texture_idx = texture_map[int(texture_part)]
                else:
                    new_texture_idx = ''
                
                if normal_part:
                    new_normal_idx = normal_map[int(normal_part)]
                else:
                    new_normal_idx = ''
                
                new_face_str.append(f"{new_vertex_idx}/{new_texture_idx}/{new_normal_idx}")
            
            f.write(f"f {' '.join(new_face_str)}\n")

def main():
    input_file = r'd:\code\warp_SPH\data\models\koifish.obj'
    
    print("正在读取OBJ文件...")
    vertices, texture_coords, normals, faces, material_lib, material = read_obj_file(input_file)
    
    print(f"读取完成: {len(vertices)}个顶点, {len(texture_coords)}个纹理坐标, {len(normals)}个法线, {len(faces)}个面")
    
    print("正在分割模型...")
    body_vertex_indices, tail_vertex_indices, body_faces, tail_faces = split_model(vertices, texture_coords, normals, faces)
    
    print(f"分割完成:")
    print(f"  鱼身: {len(body_vertex_indices)}个顶点, {len(body_faces)}个面")
    print(f"  鱼尾: {len(tail_vertex_indices)}个顶点, {len(tail_faces)}个面")
    
    print("正在保存鱼身模型...")
    body_output = r'd:\code\warp_SPH\data\models\koifish_body.obj'
    reindex_and_write_obj(body_output, body_vertex_indices, body_faces, vertices, texture_coords, normals, material_lib, material)
    
    print("正在保存鱼尾模型...")
    tail_output = r'd:\code\warp_SPH\data\models\koifish_tail.obj'
    reindex_and_write_obj(tail_output, tail_vertex_indices, tail_faces, vertices, texture_coords, normals, material_lib, material)
    
    print("分割完成!")
    print(f"鱼身模型已保存到: {body_output}")
    print(f"鱼尾模型已保存到: {tail_output}")

if __name__ == "__main__":
    main()
