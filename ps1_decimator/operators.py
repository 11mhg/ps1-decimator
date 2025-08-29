import bpy
import bmesh
import numpy as np
import heapq
import traceback

from mathutils import Vector

from .materials import replace_material_with_downsampled

def tri_area(a: Vector, b: Vector, c: Vector) ->  float:
    return ((b - a).cross(c-a)).length * 0.5

def safe_dot(a: Vector, b: Vector) -> float:
    d = max(-1.0, min(1.0, a.normalized().dot(b.normalized())))
    return d

class OBJECT_OT_ps1_decimate(bpy.types.Operator):
    """Applies a stylistic decimation to the active mesh"""
    bl_idname = "object.ps1_decimate"
    bl_label = "PS1 Decimator"
    bl_options = { "REGISTER", "UNDO" }
    
    poly_count_target: bpy.props.IntProperty(
        name="Poly Count Target",
        description = "The number of polygons to keep after decimation.",
        default=1000,
        min=1,
        max=100000,
    ) # type: ignore
    
    fixed_point_precision_bits: bpy.props.IntProperty(
        name="Fixed-Point Precision in Bits",
        description = "Precision of vertex coordinates (the number of bits to quantize vertex coordinates to)",
        default=12,
        min=1,
        max=32,
    ) # type: ignore
    
    saliency_factor: bpy.props.FloatProperty(
        name="Saliency Factor",
        description = "Weight of saliency in the cost calculation (0=QEM only, 1=Saliency only)",
        default=0.75,
        min=0.0,
        max=1.0,
    ) # type: ignore
    
    angle_factor: bpy.props.FloatProperty(
        name="Angle Factor",
        description = "A factor to weight angles in the cost calculation (0=ignore angles, 1=weigh angles heavily)",
        default=0.75,
        min=0.0,
        max=1.0,
    ) # type: ignore
    
    tex_size: bpy.props.FloatProperty(
        name="Texture Size",
        description = "The target texture size after decimation",
        default=128,
        min=1,
        max=1024,
    ) # type: ignore
    
    texture_export_path: bpy.props.StringProperty(
        name="Export Directory",
        description = "The directory to export the downsampled textures",
        default="",
        subtype="DIR_PATH",
    ) # type: ignore
    
    batch_size: bpy.props.IntProperty(
        name="Batch Size",
        description="Number of edges collapsed per modal tick",
        default=128,
        min=64,
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
            vertex_quadrics[v] = np.zeros((4, 4), dtype=np.float64)
        for f in bm.faces:
            p = self._plane_from_face(f)
            if p is None:
                continue
            K = np.outer(p, p)  # 4x4
            for v in f.verts:
                vertex_quadrics[v] += K
        return vertex_quadrics
    
    def _plane_from_face(self, f: Vector):
        n = Vector(f.normal)
        if n.length < 1e-9:
            return None
        n = n.normalized()
        p0 = f.verts[0].co
        d = -n.dot(p0)
        return np.array([n.x, n.y, n.z, d], dtype=np.float32)
    
    def _optimal_contraction(self, v1, v2, Qmap):
        Q = Qmap.get(v1, np.zeros((4, 4))) + Qmap.get(v2, np.zeros((4, 4)))
        
        A = Q.copy()
        A[3, :] = np.array([0., 0., 0., 1.], dtype=np.float32)
        b = np.array([0., 0., 0., 1.], dtype=np.float32)
        
        try:
            v_bar_h = np.linalg.Solve(A, b)
            if not np.all(np.isfinite(v_bar_h)):
                raise np.linalg.LinAlgError()
            v_bar = Vector((float(v_bar_h[0]), float(v_bar_h[1]), float(v_bar_h[2])))
            cost = float(v_bar_h @ (Q @ v_bar_h.T))
            return v_bar, cost
        except Exception:
            candidates = []
            p1 = v1.co 
            p2 = v2.co 
            mid = (p1 + p2) * 0.5
            for cand in (p1, p2, mid):
                vh = np.array([cand.x, cand.y, cand.z, 1.0], dtype=np.float32)
                c = float(vh @ (Q @ vh.T))
                candidates.append((Vector(cand), c))
            
            v_bar, cost = min(candidates, key=lambda x: x[1])
            return v_bar, float(cost)
        
        

    def compute_edge_cost(self, e, vertex_quadrics):
        v1, v2 = e.verts
        
        if e.is_boundary or not e.is_manifold:
            return float('inf')
        
        v_bar, cost = self._optimal_contraction(v1, v2, vertex_quadrics)
        return cost
        
        
        # Q = vertex_quadrics[v1] + vertex_quadrics[v2]
        # v_pos = np.append((v1.co + v2.co) / 2.0, 1.0)
        # qem_cost = v_pos @ Q @ v_pos.T

        # saliency_v1 = self.vertex_curvatures.get(v1, 0.0)
        # saliency_v2 = self.vertex_curvatures.get(v2, 0.0)
        # saliency_cost = max(saliency_v1, saliency_v2)

        # angle_bias = 1.0
        # if len(e.link_faces) >= 2:
        #     try:
        #         f1, f2 = e.link_faces
        #         dot = np.clip(np.dot(f1.normal, f2.normal), -1.0, 1.0)
        #         angle = np.arccos(dot)
        #         angle_bias = 1.0 + (angle / np.pi) * self.angle_factor
        #     except Exception as exc:
        #         print(exc)
        #         traceback.print_exc()
                

        # return ((1.0 - self.saliency_factor) * qem_cost + self.saliency_factor * saliency_cost) * angle_bias    
    
    #
    # --- Operator lifecycle ---
    #
    def invoke(self, context, event):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Active object is not a mesh")
            return {"CANCELLED"}
        
        # Read settings from scene properties
        scene = context.scene
        self.poly_count_target = scene.poly_count_target
        self.fixed_point_precision_bits = scene.fixed_point_precision_bits
        self.saliency_factor = scene.saliency_factor
        self.angle_factor = scene.angle_factor
        self.tex_size = scene.tex_size
        self.texture_export_path = scene.texture_export_path
        self.batch_size = scene.batch_size
        scene.is_decimating = True
        
        self.obj = obj
        self.bm = bmesh.new()
        self.bm.from_mesh(obj.data)
        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()
        
        print(f"PS1 Decimator: Starting with {len(self.bm.faces)} faces, target: {self.poly_count_target}")
        print(f"PS1 Decimator: Batch size: {self.batch_size}")

        self._precompute_all_vertex_curvatures(self.bm)
        self.vertex_quadrics = self.compute_vertex_quadrics(self.bm)
        
        print(f"PS1 Decimator: Vertex quadrics computed")

        self.target_count = min(self.poly_count_target, len(self.bm.faces))
        self.start_faces = len(self.bm.faces)

        self.calculate_edge_queue()

        self._timer = context.window_manager.event_timer_add(0.01, window=context.window)
        context.window_manager.modal_handler_add(self)
        
        self.report({'INFO'}, f"Starting decimation: {self.start_faces} faces → {self.target_count} target")
        self.batch_count = 0
        return {"RUNNING_MODAL"}

    def push_edge(self, edge, cost):
        if edge.is_valid and edge not in self.in_queue:
            heapq.heappush(self.edge_queue, (cost, id(edge), edge))
            self.in_queue.add(edge)

    def modal(self, context, event):
        if event.type == "ESC":
            self.cleanup(cancel=True, context=context)
            return {"CANCELLED"}
        
        if event.type == "TIMER":
            self.batch_count += 1
            print(f"PS1 Decimator: Processing batch, current faces: {len(self.bm.faces)}, target: {self.target_count}")
            context.scene.is_decimating = True
            current_target_count = len(self.bm.faces) - self.batch_size
            while len(self.bm.faces) > current_target_count:
                batch = self.get_independent_edge_batch(self.batch_size)
                
                if len(self.bm.faces) <= self.target_count or not self.edge_queue or not batch:
                    self.cleanup(cancel=False, context=context)
                    return {"FINISHED"}
                
                try:
                    #bmesh.ops.collapse(self.bm, edges=batch, uvs=True)
                    bmesh.ops.dissolve_edges(self.bm, edges=batch, use_verts=True, use_face_split=True)
                except Exception as e:
                    print(f"Failed to collapse edge: {e}")
                    traceback.print_exc()
                    continue
            
            self.cleanup_pass()
            #self.quadriflow_pass()
            
            # Update progress reporting
            done = self.start_faces - len(self.bm.faces)
            total = self.start_faces - self.target_count
            if total > 0:
                progress = float(min(100, int((done / total) * 100)))
                context.scene.decimation_progress = progress
                self.report({'INFO'}, f"Decimating... {progress}% ({len(self.bm.faces)}/{self.target_count} faces)")
        
        return {"RUNNING_MODAL"}
    
    def quadriflow_pass(self):
        if self.batch_count % 100 != 0:
            return
        
        target_faces = len(self.bm.faces)
        
        self.bm.from_mesh(self.obj.data)
        self.obj.data.update()
        self.bm.free()
        
        
        bpy.ops.object.quadriflow_remesh(
            target_faces=target_faces,
            use_preserve_sharp=True,
            use_mesh_symmetry=False,
            use_preserve_boundary=True,
        )
        
        self.bm = bmesh.new()
        self.bm.from_mesh(self.obj.data)
        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()
        
        self.vertex_quadrics = self.compute_vertex_quadrics(self.bm)
        self.calculate_edge_queue()
        
        
    def cleanup_pass(self):
        if self.batch_count % 100 != 0:
            return
        # merge stray tris into quads
        bmesh.ops.join_triangles(
            self.bm,
            faces=self.bm.faces,
            angle_face_threshold=0.01,
            angle_shape_threshold=1.0
        )
        # smooth vertex distribution
        # bmesh.ops.smooth_vert(self.bm, verts=self.bm.verts, factor=0.3)
        # remove doubles
        bmesh.ops.remove_doubles(self.bm, verts=self.bm.verts, dist=1e-6)
        # recalc normals
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)
        
        #self.calculate_edge_queue()
    
    def calculate_edge_queue(self):
        self.edge_queue = []
        self.in_queue = set()

        for edge in self.bm.edges:
            cost = self.compute_edge_cost(edge, self.vertex_quadrics)
            self.push_edge(edge, cost)
        print(f"PS1 Decimator: Edge queue built with {len(self.edge_queue)} edges")
    
    def get_independent_edge_batch(self, max_batch_size=128):
        """Get a batch of edges that don't share vertices"""
        batch = []
        used_vertices = set()
        
        temp_queue = []
        
        # Extract edges from queue until we have a safe batch
        while len(batch) < max_batch_size and self.edge_queue:
            cost, edge_id, edge = heapq.heappop(self.edge_queue)
            self.in_queue.discard(edge)
            
            if not edge.is_valid:
                continue
            
            
            if len(edge.verts) != 2:
                continue
                
            v1, v2 = edge.verts
            
            # Check if vertices are already used in this batch
            if v1 not in used_vertices and v2 not in used_vertices:
                batch.append(edge)
                used_vertices.add(v1)
                used_vertices.add(v2)
            else:
                # Put back in queue for later
                temp_queue.append((cost, edge))
        
        # Put unused edges back in queue
        for cost, edge in temp_queue:
            if edge.is_valid:
                heapq.heappush(self.edge_queue, (cost, id(edge), edge))
                self.in_queue.add(edge)
        
        return batch

    
    def cleanup(self, cancel, context):
        """Clean up modal operation"""
        print(f"PS1 Decimator: Cleanup called, cancel={cancel}, final faces: {len(self.bm.faces)}")
        context.scene.is_decimating = False
        context.scene.decimation_progress = 100
        context.window_manager.event_timer_remove(self._timer)
        
        if cancel:
            self.report({'INFO'}, "Decimation cancelled")
            self.bm.free()
            return
        
        # Apply vertex quantization (PS1 fixed-point precision)
        scale_factor = 2 ** self.fixed_point_precision_bits
        coords = np.array([v.co[:] for v in self.bm.verts], dtype=np.float32)
        coords = np.round(coords * scale_factor) / scale_factor
        for v, c in zip(self.bm.verts, coords):
            v.co = c

        # Process materials and textures
        for slot in self.obj.material_slots:
            if slot.material:
                replace_material_with_downsampled(
                    slot.material, 
                    self.tex_size, 
                    export_path=self.texture_export_path if self.texture_export_path else None
                )
                
        # Remove doubles and recalc normals
        bmesh.ops.remove_doubles(self.bm, verts=self.bm.verts, dist=1e-6)
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)

        # Apply changes to mesh
        end_faces = len(self.bm.faces)
        self.bm.to_mesh(self.obj.data)
        self.bm.free()
        self.obj.data.update()
        
        # Apply flat shading for PS1 look
        bpy.ops.object.shade_flat()
        
        # Final report
        self.report({'INFO'}, f"PS1 Decimation Complete! Faces: {self.start_faces} → {end_faces} (−{self.start_faces - end_faces})")
        
def register():
    bpy.utils.register_class(OBJECT_OT_ps1_decimate)
    
def unregister():
    bpy.utils.unregister_class(OBJECT_OT_ps1_decimate)