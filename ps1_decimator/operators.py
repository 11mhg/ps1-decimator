import bpy
import bmesh
import numpy as np
import heapq

from mathutils import Vector

from .materials import replace_material_with_downsampled


class OBJECT_OT_ps1_decimate(bpy.types.Operator):
    """Applies a stylistic decimation to the active mesh"""
    bl_idname = "object.ps1_decimate"
    bl_label = "PS1 Decimator"
    bl_options = { "REGISTER", "UNDO" }
    
    ratio: bpy.props.FloatProperty(
        name="Decimation Ratio",
        description = "The fraction of faces to keep",
        default=0.2,
        min=1e-1,
        max=1.0,
    ) # type: ignore
    
    scale_factor: bpy.props.FloatProperty(
        name="Scale Factor",
        description = "The Scaling Factor the for the fixed precision grid",
        default=4096,
        min=1.0,
        max=8192.0,
    ) # type: ignore
    
    uv_steps: bpy.props.FloatProperty(
        name="UV Quantization",
        description = "Number of steps for UVs (higher = smoother, lower = blocky)",
        default=16,
        min=2,
        max=512,
    ) # type: ignore
    
    saliency_factor: bpy.props.FloatProperty(
        name="Saliency Factor",
        description = "Weight of saliency in the cost calculation (0=QEM only, 1=Saliency only)",
        default=0.5,
        min=0.0,
        max=1.0,
    ) # type: ignore
    
    angle_factor: bpy.props.FloatProperty(
        name="Angle Factor",
        description = "A factor to weight angles in the cost calculation (0=ignore angles, 1=weigh angles heavily)",
        default=0.0,
        min=0.0,
        max=1.0,
    ) # type: ignore
    
    tex_size: bpy.props.FloatProperty(
        name="Texture Size",
        description = "The target texture size after decimation",
        default=256,
        min=1,
        max=1024,
    ) # type: ignore
    
    def _precompute_all_vertex_curvatures(self, bm):
        self.vertex_curvatures = {}
        raw_curvatures = {}
        
        for v in bm.verts:
            normal_sum = sum((f.normal.normalized() * f.calc_area() for f in v.link_faces), Vector((0.0, 0.0, 0.0)))
            raw_curvatures[v] = 0.0
            if len(v.link_faces) > 0 and v.normal.length > 1e-6:
                avg_normal = normal_sum.normalized()
                raw_curvatures[v] = 1.0 - np.dot(np.array(v.normal), np.array(avg_normal))
        
        if raw_curvatures:
            all_values = list(raw_curvatures.values())
            min_curv = min(all_values)
            max_curv = max(all_values)
            
            if max_curv - min_curv > 1e-6:
                for v, curv in raw_curvatures.items():
                    self.vertex_curvatures[v] = (curv - min_curv) / (max_curv - min_curv)
            else:
                for v in bm.verts:
                    self.vertex_curvatures[v] = 0.0
                    
    
    def compute_vertex_quadrics(self, bm):
        vertex_quadrics = {}
        for v in bm.verts:
            Q = np.zeros((4, 4))
            for f in v.link_faces:
                normal = f.normal
                d = -np.dot(f.calc_center_median(), normal)
                Q += self.compute_face_quadric(normal, d)
            vertex_quadrics[v] = Q        
        return vertex_quadrics
    
    def compute_face_quadric(self,normal, d=0.0):
        n = np.array(normal).reshape(3, 1)
        Q = np.zeros((4, 4))
        Q[:3, :3] = n @ n.T
        Q[:3, 3] = n.flatten() * d
        Q[3, :3] = n.flatten() * d
        Q[3,3] = d*d
        return Q
    
    def compute_edge_cost(self, e, vertex_quadrics):
        v1, v2 = e.verts
        Q = vertex_quadrics[v1] + vertex_quadrics[v2]
        v_pos = np.append((v1.co + v2.co) / 2.0, 1.0)
        qem_cost = v_pos @ Q @ v_pos.T
        
        # mesh saliency component
        saliency_v1 = self.vertex_curvatures.get(v1, 0.0)
        saliency_v2 = self.vertex_curvatures.get(v2, 0.0)
        saliency_cost = max(saliency_v1, saliency_v2)
        
        angle_bias = 1.0 
        if len(e.link_faces) > 2:
            f1, f2 = e.link_faces
            dot = np.clip(np.dot(f1.normal, f2.normal), -1.0, 1.0)
            angle = np.arccos(dot)
            
            angle_bias = 1.0 + (angle / np.pi) * self.angle_factor
        
        cost = ((1.0 - self.saliency_factor) * qem_cost + self.saliency_factor * saliency_cost) * angle_bias
        return cost
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Active object is not a mesh")
            return {"CANCELLED"}
        
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        #bmesh.ops.triangulate(bm, faces=bm.faces)
        
        self._precompute_all_vertex_curvatures(bm)
        vertex_quadrics = self.compute_vertex_quadrics(bm)
        
        target_count = int(len(bm.faces) * self.ratio)
        
        edge_queue = []
        for edge in bm.edges:
            cost = self.compute_edge_cost(edge, vertex_quadrics)
            heapq.heappush(edge_queue, (cost, edge.index))
        
        while len(bm.faces) > target_count and edge_queue:
            cost, edge_idx = heapq.heappop(edge_queue)
            
            if np.isinf(cost):
                break
            
            try:
                edge = bm.edges[edge_idx]
                if not edge.is_valid:
                    continue
                
                v1, v2 = edge.verts
                ret = bmesh.ops.collapse_edge(bm, edge=edge)
                
                new_vert = ret['verts']
                
                new_quadric = vertex_quadrics[v1] + vertex_quadrics[v2]
                vertex_quadrics[new_vert] = new_quadric
                
                for e_affected in new_vert.link_edges:
                    new_cost = self.compute_edge_cost(e_affected, vertex_quadrics)
                    heapq.heappush(edge_queue, (new_cost, e_affected.index))
            except Exception as e:
                continue
        
        # vertex snapping
        coords = np.array([v.co[:] for v in bm.verts], dtype=np.float32)
        coords = np.round(coords * self.scale_factor) / self.scale_factor
        for v, c in zip(bm.verts, coords):
            v.co = c
        
        # uv snapping
        for slot in obj.material_slots:
            if slot.material:
                replace_material_with_downsampled(slot.material, self.tex_size)
        
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        bpy.ops.object.shade_flat()
        
        return {"FINISHED"}
        
def register():
    bpy.utils.register_class(OBJECT_OT_ps1_decimate)
    
def unregister():
    bpy.utils.unregister_class(OBJECT_OT_ps1_decimate)