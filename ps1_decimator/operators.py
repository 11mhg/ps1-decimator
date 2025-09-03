"""
PS1 Decimator operator using Open3D.

This module defines an operator that decimates the active Blender mesh to a
target number of faces using Open3D's quadratic error metric (QEM)
simplification.  After decimation it quantizes vertex positions to a
fixed‑point grid (for a characteristic PS1 look), optionally passes the
result through Blender's QuadriFlow remesher, downsamples all textures on
the object's materials, and generates a displaced subdivision mesh that
snaps a subdivided version of the decimated mesh back to the original
surface using nearest‑point queries.  All trimesh references have been
removed; Open3D is used exclusively for geometry processing.  A fallback
to Blender's decimate modifier is provided when Open3D is unavailable.
"""

from __future__ import annotations

import bpy
import numpy as np

try:
    import open3d as o3d  # type: ignore
    _HAS_O3D = True
except Exception:
    o3d = None  # type: ignore
    _HAS_O3D = False

from mathutils import Vector
from typing import Optional

from .materials import replace_material_with_downsampled


# -----------------------------------------------------------------------------
# Blender ↔ Open3D conversion helpers
# -----------------------------------------------------------------------------

def mesh_to_open3d(obj: bpy.types.Object) -> "o3d.geometry.TriangleMesh":
    """Convert a Blender mesh object to an Open3D TriangleMesh.

    The mesh is evaluated and triangulated via loop triangles to ensure a
    purely triangular representation (Open3D cannot handle ngons directly).
    The resulting Open3D mesh is returned with vertex normals computed.

    Parameters
    ----------
    obj : bpy.types.Object
        A Blender object of type 'MESH'.

    Returns
    -------
    o3d.geometry.TriangleMesh
        A triangulated Open3D mesh representing the input object.  If
        Open3D is unavailable, raises RuntimeError.
    """
    if not _HAS_O3D:
        raise RuntimeError("Open3D not available")
    # Evaluate the object to apply modifiers and deformations
    me_eval = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    try:
        me_eval.calc_loop_triangles()
        verts = np.array([v.co[:] for v in me_eval.vertices], dtype=np.float64)
        tris = np.array([lt.vertices[:] for lt in me_eval.loop_triangles], dtype=np.int32)
        mesh_o3 = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(tris),
        )
        if not mesh_o3.has_vertex_normals():
            mesh_o3.compute_vertex_normals()
        return mesh_o3
    finally:
        obj.to_mesh_clear()


def open3d_to_mesh(o3_mesh: "o3d.geometry.TriangleMesh", mesh: bpy.types.Mesh) -> None:
    """Write an Open3D TriangleMesh back into a Blender Mesh datablock.

    The target Blender mesh is cleared and rebuilt using ``from_pydata``.
    After geometry assignment, ``mesh.validate`` and ``mesh.update`` are
    called to rebuild edges and ensure data consistency.  Normals are not
    explicitly recalculated with ``calc_normals`` because that API has been
    removed in Blender 4.5; instead, ``validate`` and ``update`` suffice.

    Parameters
    ----------
    o3_mesh : o3d.geometry.TriangleMesh
        The Open3D mesh whose vertex and triangle data will populate the
        Blender mesh.
    mesh : bpy.types.Mesh
        The Blender mesh datablock to write into.
    """
    verts = np.asarray(o3_mesh.vertices, dtype=np.float32)
    tris = np.asarray(o3_mesh.triangles, dtype=np.int32)
    mesh.clear_geometry()
    if verts.size == 0 or tris.size == 0:
        mesh.update()
        return
    # Build mesh via from_pydata to avoid touching read‑only attributes
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in tris])
    # Validate and update
    mesh.validate(clean_customdata=True)
    mesh.update(calc_edges=True, calc_edges_loose=True)
    # Choose flat shading for PS1 aesthetic
    for p in mesh.polygons:
        # Disable smoothing on each polygon for a flat PS1 aesthetic
        p.use_smooth = False
    # Some Blender versions (e.g. 4.5) remove the `use_auto_smooth` attribute from
    # Mesh; skipping assignment here avoids an AttributeError.  If this
    # attribute exists on the mesh datablock, developers may explicitly set it
    # to False to disable split normals.
    if hasattr(mesh, "use_auto_smooth"):
        setattr(mesh, "use_auto_smooth", False)


def quantize_vertices_o3d(o3_mesh: "o3d.geometry.TriangleMesh", bits: int) -> None:
    """Quantize vertex positions to a fixed‑point grid.

    Each coordinate is rounded to a multiple of 1/2**bits.  This imparts
    the characteristic wobbly PS1 look.  The mesh is modified in place.

    Parameters
    ----------
    o3_mesh : o3d.geometry.TriangleMesh
        The mesh to quantize.
    bits : int
        Number of fractional bits; the grid spacing is 1/(2**bits).
    """
    if bits <= 0:
        return
    scale = float(1 << bits)
    v = np.asarray(o3_mesh.vertices, dtype=np.float64)
    v = np.round(v * scale) / scale
    o3_mesh.vertices = o3d.utility.Vector3dVector(v)


# -----------------------------------------------------------------------------
# Directional projection helper
# -----------------------------------------------------------------------------
def _project_points_directional(
    reference: "o3d.geometry.TriangleMesh",
    points: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Project query points onto a reference mesh along specified normals.

    This function shoots a ray from each point in the direction of its
    corresponding normal and also in the opposite direction.  For each point
    the nearest valid hit (smallest positive distance) is selected.  If no
    intersection is found, the closest surface point (via `_closest_points_raycast`)
    is used instead.  When Open3D's tensor API is unavailable, a KDTree
    fallback returns the nearest vertex.

    Parameters
    ----------
    reference : o3d.geometry.TriangleMesh
        The mesh onto which to project the points.
    points : np.ndarray
        Nx3 array of query points.
    normals : np.ndarray
        Nx3 array of unit normals associated with each query point.

    Returns
    -------
    np.ndarray
        Nx3 array of projected points.
    """
    try:
        import open3d.t as o3dt  # type: ignore
        # Construct tensor mesh for raycasting
        tverts = o3dt.geometry.Tensor(
            np.asarray(reference.vertices), dtype=o3dt.core.Dtype.Float32
        )
        ttris = o3dt.geometry.Tensor(
            np.asarray(reference.triangles), dtype=o3dt.core.Dtype.Int32
        )
        tmesh = o3dt.geometry.TriangleMesh(tverts, ttris)
        scene = o3dt.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)

        # Normalize normals to avoid zero-length directions
        norms = normals.copy().astype(np.float32)
        lengths = np.linalg.norm(norms, axis=1, keepdims=True)
        norms /= lengths + 1e-12

        # Cast forward and backward rays
        rays_fwd = np.hstack([points.astype(np.float32), norms])
        rays_bwd = np.hstack([points.astype(np.float32), -norms])
        res_fwd = scene.cast_rays(o3dt.core.Tensor(rays_fwd))
        res_bwd = scene.cast_rays(o3dt.core.Tensor(rays_bwd))
        t_hit_fwd = res_fwd["t_hit"].numpy().reshape(-1)
        t_hit_bwd = res_bwd["t_hit"].numpy().reshape(-1)

        proj = np.empty_like(points)
        fallback_idx = []  # indices requiring fallback
        for i in range(points.shape[0]):
            candidates = []
            # Forward hit
            tf = t_hit_fwd[i]
            if np.isfinite(tf) and tf > 1e-6:
                candidates.append(points[i] + norms[i] * tf)
            # Backward hit
            tb = t_hit_bwd[i]
            if np.isfinite(tb) and tb > 1e-6:
                candidates.append(points[i] - norms[i] * tb)
            if candidates:
                # Choose the candidate closest to the original point
                dists = [np.linalg.norm(c - points[i]) for c in candidates]
                proj[i] = candidates[int(np.argmin(dists))]
            else:
                # Defer to fallback if no candidate
                fallback_idx.append(i)
        # Fallback to closest surface point for unresolved points
        if fallback_idx:
            fallback_points = points[fallback_idx]
            fallback_proj = _closest_points_raycast(reference, fallback_points)
            proj[fallback_idx] = fallback_proj
        return proj
    except Exception:
        # Fallback: use nearest vertex via KDTree for all points
        return _closest_points_raycast(reference, points)

# -----------------------------------------------------------------------------
# Displacement generation
# -----------------------------------------------------------------------------

def _closest_points_raycast(
    target: "o3d.geometry.TriangleMesh", points: np.ndarray
) -> np.ndarray:
    """Compute closest points on the target surface for each query point.

    Tries to use Open3D's RaycastingScene (requires ``open3d.t``) for
    accurate surface projection.  If unavailable, falls back to nearest
    vertex lookup via KDTree, which approximates the closest point by the
    nearest vertex on the target mesh.

    Parameters
    ----------
    target : o3d.geometry.TriangleMesh
        Mesh representing the surface to project onto.
    points : np.ndarray
        Nx3 array of query points.

    Returns
    -------
    np.ndarray
        Nx3 array of closest points.
    """
    # Try raycasting (Open3D >= 0.15)
    try:
        import open3d.t as o3dt  # type: ignore
        tverts = o3dt.geometry.Tensor(np.asarray(target.vertices), dtype=o3dt.core.Dtype.Float32)
        ttris = o3dt.geometry.Tensor(np.asarray(target.triangles), dtype=o3dt.core.Dtype.Int32)
        tmesh = o3dt.geometry.TriangleMesh(tverts, ttris)
        scene = o3dt.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)
        q = o3dt.core.Tensor(points.astype(np.float32))
        out = scene.compute_closest_points(q)
        return out["points"].numpy()
    except Exception:
        # Fallback: KDTree on vertices
        kdtree = o3d.geometry.KDTreeFlann(target)
        V = np.asarray(target.vertices, dtype=np.float64)
        closest = np.empty_like(points)
        for i, p in enumerate(points):
            _, idx, _ = kdtree.search_knn_vector_3d(p, 1)
            closest[i] = V[idx[0]] if len(idx) else p
        return closest


def generate_displaced_subdivision_open3d(
    base: "o3d.geometry.TriangleMesh",
    reference: "o3d.geometry.TriangleMesh",
    levels: int = 1,
    apply_displacement: bool = True,
    directional_projection: bool = False,
) -> "o3d.geometry.TriangleMesh":
    """Create a displaced subdivision mesh from a decimated base.

    The base mesh is subdivided ``levels`` times using midpoint
    subdivision.  Each vertex of the subdivided mesh is projected onto
    the reference surface.  If ``directional_projection`` is True, the
    projection is computed by casting rays along the subdivided vertex
    normals (both forward and backward) using Open3D's raycasting
    system.  Otherwise, a closest–point query via
    ``_closest_points_raycast`` is used.  The projection is applied to
    the subdivided mesh when ``apply_displacement`` is True; otherwise
    the computed positions are ignored and only the subdivided mesh is
    returned.

    Parameters
    ----------
    base : o3d.geometry.TriangleMesh
        The decimated mesh to subdivide.
    reference : o3d.geometry.TriangleMesh
        The original mesh used for computing projection targets.
    levels : int, optional
        Number of subdivision iterations.  At least 1.
    apply_displacement : bool, optional
        If True, overwrite the subdivided vertices with their projected
        positions.
    directional_projection : bool, optional
        If True, project points along their vertex normals; if False,
        project to the nearest surface point.

    Returns
    -------
    o3d.geometry.TriangleMesh
        The subdivided mesh; vertices may be displaced if
        ``apply_displacement`` is True.
    """
    if levels < 1:
        levels = 1
    mesh = base
    for _ in range(levels):
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    if directional_projection:
        # Ensure normals are up to date
        mesh.compute_vertex_normals()
        norms = np.asarray(mesh.vertex_normals, dtype=np.float64)
        proj_pts = _project_points_directional(reference, pts, norms)
    else:
        proj_pts = _closest_points_raycast(reference, pts)
    if apply_displacement:
        mesh.vertices = o3d.utility.Vector3dVector(proj_pts)
    mesh.compute_vertex_normals()
    return mesh


# -----------------------------------------------------------------------------
# Operator class
# -----------------------------------------------------------------------------

class OBJECT_OT_ps1_decimate(bpy.types.Operator):
    """Decimate the active mesh with Open3D and build a displaced subdivision."""

    bl_idname = "object.ps1_decimate"
    bl_label = "PS1 Decimator (Open3D)"
    bl_options = {"REGISTER", "UNDO"}

    # Operator properties
    poly_count_target: bpy.props.IntProperty(
        name="Target Faces (tri)",
        default=5000,
        min=10,
        soft_max=200000,
        description="Target number of triangles after decimation",
    )  # type: ignore

    fixed_point_precision_bits: bpy.props.IntProperty(
        name="PS1 Grid Bits",
        default=8,
        min=0,
        max=16,
        description="Number of fractional bits for vertex quantization",
    )  # type: ignore

    use_quadriflow_remesh: bpy.props.BoolProperty(
        name="QuadriFlow (optional)",
        default=False,
        description="Run Blender's QuadriFlow remesher on the decimated mesh",
    )  # type: ignore

    tex_size: bpy.props.IntProperty(
        name="Texture Size",
        default=256,
        min=8,
        soft_max=2048,
        description="Downsample textures to approximately this size",
    )  # type: ignore

    texture_export_path: bpy.props.StringProperty(
        name="Export PNGs To",
        subtype="DIR_PATH",
        default="",
        description="Optional folder to save downsampled textures",
    )  # type: ignore

    # Unused properties kept for UI compatibility
    saliency_factor: bpy.props.FloatProperty(default=0.0)  # type: ignore
    angle_factor: bpy.props.FloatProperty(default=0.0)  # type: ignore
    param_factor: bpy.props.FloatProperty(default=0.0)  # type: ignore
    feature_factor: bpy.props.FloatProperty(default=0.0)  # type: ignore

    # Toggle to use vertex clustering remeshing instead of QEM decimation.
    # When enabled, the mesh is simplified by pooling vertices into a voxel
    # grid (vertex clustering) and then each new vertex is projected back
    # onto the original surface via the closest point operator.  This can
    # produce a different aesthetic and may better preserve thin features
    # when combined with the automatic or manual protection masks.
    use_vertex_clustering_remesh: bpy.props.BoolProperty(
        name="Use Vertex Clustering",
        description=(
            "Simplify mesh via vertex clustering rather than QEM decimation. "
            "Vertices are clustered into a voxel grid and then snapped to the "
            "closest points on the original mesh."
        ),
        default=False,
    )  # type: ignore

    # Scaling factor for voxel size used during vertex clustering.  The
    # clustering voxel size is computed as ``bbox_diagonal / cbrt(target)``
    # multiplied by this factor.  Larger values result in coarser meshes.
    vertex_clustering_factor: bpy.props.FloatProperty(
        name="Clustering Factor",
        description=(
            "Scale factor for the voxel size in vertex clustering. "
            "Increase to simplify more aggressively; decrease to retain detail."
        ),
        default=1.0,
        min=0.01,
        max=10.0,
    )  # type: ignore

    # Name of a vertex group whose vertices should be preserved.  If this
    # property is non‑empty and a matching group exists on the active object,
    # those vertices will be kept intact during decimation.  This option
    # complements the automatic feature preservation controlled by
    # ``saliency_factor`` and ``feature_factor`` – if a vertex group is
    # specified, it takes precedence over the automatic thresholds.  Use
    # vertex groups to manually mark critical areas such as fingers, joints
    # or facial features that must remain unaltered.
    protected_vertex_group: bpy.props.StringProperty(
        name="Protected Vertex Group",
        description=(
            "Name of vertex group to preserve during decimation. "
            "When empty, automatic preservation can be enabled via "
            "saliency and feature factors."
        ),
        default="",
    )  # type: ignore

    # Toggle to control whether subdivided vertices are projected along their
    # normals or simply snapped to the closest surface point.  When enabled,
    # each vertex of the subdivided mesh casts rays in the direction of its
    # normal (and its inverse) and chooses the first valid hit on the
    # reference surface.  This helps prevent vertices from snapping to
    # unrelated surfaces in the presence of nearby geometry (e.g. thin
    # platforms above a lower floor).  When disabled, vertices are projected
    # to the nearest point in Euclidean space.
    use_directional_projection: bpy.props.BoolProperty(
        name="Directional Projection",
        description=(
            "Project subdivided vertices along their normals when snapping back "
            "to the original surface.  This reduces snapping onto nearby but "
            "unrelated surfaces."
        ),
        default=True,
    )  # type: ignore

    # Directional projection toggle.  When enabled, the displaced subdivision
    # projects vertices along the subdivided mesh normals (forward and
    # backward) onto the original mesh.  This reduces artifacts where
    # vertices snap to nearby surfaces that are not visible along the
    # vertex normal direction.  If disabled, projection uses the
    # nearest surface point regardless of direction.
    use_directional_projection: bpy.props.BoolProperty(
        name="Directional Projection",
        description=(
            "Project subdivided vertices along normals when generating the "
            "displaced mesh; improves fidelity of thin features and reduces "
            "snapping to distant surfaces"
        ),
        default=False,
    )  # type: ignore

    def execute(self, context: bpy.types.Context) -> set[str]:
        """Execute the PS1 decimation operator.

        The operator supports three modes of operation depending on the
        availability of Open3D and user settings:

        1. **Protected decimation**: If Open3D is available and either a
           protected vertex group is specified or the saliency/feature
           thresholds are non‑zero, vertices deemed important are kept
           intact while the rest of the mesh is simplified.  Importance can
           be defined manually via a vertex group or automatically via
           curvature and local edge length thresholds.
        2. **Standard Open3D decimation**: If Open3D is available but no
           protected group or thresholds are provided, the entire mesh is
           decimated uniformly.
        3. **Blender decimate modifier fallback**: If Open3D is not
           available, Blender's built‑in decimate modifier is used.  A
           protected vertex group is honoured in this path, preserving
           vertices assigned to the group by setting the modifier's
           ``vertex_group`` and ``vertex_group_factor`` properties【733770264643972†L3240-L3333】.
        """
        # Validate active object
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({"ERROR"}, "Select a mesh object")
            return {"CANCELLED"}

        # Duplicate original mesh for reference and for decimation operations
        original_mesh = obj.data.copy()
        original_obj = bpy.data.objects.new(name="__ps1_orig__", object_data=original_mesh)
        original_obj.matrix_world = obj.matrix_world.copy()

        # Determine whether we should preserve certain vertices.  If a
        # protected vertex group is specified, use its membership.  Otherwise,
        # if saliency_factor or feature_factor are greater than zero, compute
        # automatic importance values based on curvature and local edge
        # lengths.  When neither condition holds, all vertices are treated
        # equally.
        use_protected = False
        is_protected: Optional[np.ndarray] = None
        vg_name = self.protected_vertex_group.strip()
        # If a named group is specified, try to get its weights
        if vg_name:
            vg = obj.vertex_groups.get(vg_name)
            if vg is None:
                self.report({"ERROR"}, f"Vertex group '{vg_name}' not found on object")
                bpy.data.objects.remove(original_obj, do_unlink=True)
                return {"CANCELLED"}
            v_count = len(obj.data.vertices)
            # Build boolean mask for vertices in the group
            mask = np.zeros(v_count, dtype=bool)
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == vg.index and g.weight > 0.0:
                        mask[v.index] = True
                        break
            is_protected = mask
            use_protected = True
        # Otherwise, compute automatic importance if thresholds are non‑zero
        elif (self.saliency_factor > 0.0 or self.feature_factor > 0.0) and _HAS_O3D:
            try:
                # Convert to Open3D once to access vertices and triangles
                temp_o3 = mesh_to_open3d(original_obj)
            except Exception as e:
                self.report({"ERROR"}, f"Failed to convert mesh to Open3D: {e}")
                bpy.data.objects.remove(original_obj, do_unlink=True)
                return {"CANCELLED"}
            verts_np = np.asarray(temp_o3.vertices)
            tris_np = np.asarray(temp_o3.triangles)
            v_count = len(verts_np)
            # Compute face normals
            v0 = verts_np[tris_np[:, 0]]
            v1 = verts_np[tris_np[:, 1]]
            v2 = verts_np[tris_np[:, 2]]
            face_normals = np.cross(v1 - v0, v2 - v0)
            # Normalize normals and avoid division by zero
            fnorms = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12)
            # Build edge to faces mapping
            from collections import defaultdict
            edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
            for fi, tri in enumerate(tris_np):
                for e0, e1 in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    u, v = (int(e0), int(e1))
                    if u > v:
                        u, v = v, u
                    edge_faces[(u, v)].append(fi)
            # Initialize curvature and edge length accumulators
            curv = np.zeros(v_count, dtype=np.float64)
            edge_len_sum = np.zeros(v_count, dtype=np.float64)
            edge_len_count = np.zeros(v_count, dtype=np.int32)
            # Compute curvature contributions and edge lengths
            for (u, v), fids in edge_faces.items():
                # Edge length
                length = np.linalg.norm(verts_np[v] - verts_np[u])
                edge_len_sum[u] += length
                edge_len_sum[v] += length
                edge_len_count[u] += 1
                edge_len_count[v] += 1
                # Curvature contribution from dihedral angle if exactly two faces share edge
                if len(fids) == 2:
                    fn1, fn2 = fnorms[fids[0]], fnorms[fids[1]]
                    dot_val = float(np.clip(np.dot(fn1, fn2), -1.0, 1.0))
                    angle = np.arccos(dot_val)
                    curv[u] += angle
                    curv[v] += angle
            # Normalize curvature to [0,1]
            if curv.max() > 0:
                curv_norm = curv / curv.max()
            else:
                curv_norm = curv
            # Compute average edge length per vertex and normalize inverted lengths
            avg_len = np.zeros(v_count, dtype=np.float64)
            nonzero = edge_len_count > 0
            avg_len[nonzero] = edge_len_sum[nonzero] / edge_len_count[nonzero]
            if nonzero.any():
                min_len = avg_len[nonzero].min()
                max_len = avg_len[nonzero].max()
                if max_len > min_len:
                    # Invert: shorter edges yield higher feature value
                    feat_norm = 1.0 - (avg_len - min_len) / (max_len - min_len)
                    feat_norm = np.clip(feat_norm, 0.0, 1.0)
                else:
                    feat_norm = np.zeros_like(avg_len)
            else:
                feat_norm = np.zeros_like(avg_len)
            # Determine protected vertices via thresholds
            mask = np.zeros(v_count, dtype=bool)
            if self.saliency_factor > 0.0:
                mask |= curv_norm >= float(self.saliency_factor)
            if self.feature_factor > 0.0:
                mask |= feat_norm >= float(self.feature_factor)
            is_protected = mask
            use_protected = True
            # Free the temporary Open3D mesh
            # Note: deletion of temp_o3 is implicit; Python GC will handle
        # End of automatic importance computation

        # Case 1: Vertex clustering remesh (optional)
        # If Open3D is available and the vertex clustering switch is set,
        # perform remeshing by vertex clustering.  Protected vertices
        # (manual or automatic) are retained and not clustered.  Each new
        # vertex is projected back to the original surface to minimize
        # shrinkage.
        if _HAS_O3D and self.use_vertex_clustering_remesh:
            try:
                orig_o3 = mesh_to_open3d(original_obj)
            except Exception as e:
                self.report({"ERROR"}, f"Failed to convert mesh to Open3D: {e}")
                bpy.data.objects.remove(original_obj, do_unlink=True)
                return {"CANCELLED"}
            verts_np = np.asarray(orig_o3.vertices)
            tris_np = np.asarray(orig_o3.triangles)
            # If protected vertices are defined, partition faces
            if use_protected and is_protected is not None and is_protected.any():
                protected_face_mask = np.any(is_protected[tris_np], axis=1)
                protected_tris = tris_np[protected_face_mask]
                rest_tris = tris_np[~protected_face_mask]
                # Build protected submesh
                prot_vert_ids = np.unique(protected_tris.flatten())
                prot_vert_map = {int(old_idx): idx for idx, old_idx in enumerate(prot_vert_ids)}
                prot_vertices = verts_np[prot_vert_ids]
                prot_faces_local = np.array([[prot_vert_map[int(v)] for v in face] for face in protected_tris], dtype=np.int32)
                mesh_prot = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(prot_vertices),
                    o3d.utility.Vector3iVector(prot_faces_local),
                )
                # Build rest submesh
                if len(rest_tris) > 0:
                    rest_vert_ids = np.unique(rest_tris.flatten())
                    rest_vert_map = {int(old_idx): idx for idx, old_idx in enumerate(rest_vert_ids)}
                    rest_vertices = verts_np[rest_vert_ids]
                    rest_faces_local = np.array([[rest_vert_map[int(v)] for v in face] for face in rest_tris], dtype=np.int32)
                    mesh_rest = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(rest_vertices),
                        o3d.utility.Vector3iVector(rest_faces_local),
                    )
                else:
                    mesh_rest = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(),
                        o3d.utility.Vector3iVector(),
                    )
                target_total = int(self.poly_count_target)
                protected_count = len(protected_tris)
                target_rest = max(1, target_total - protected_count)
                # Compute voxel size: bounding box diagonal divided by cube root
                # of target face count, scaled by factor
                if len(rest_tris) > 0:
                    # Use bounding box of rest region
                    rv = np.asarray(mesh_rest.vertices)
                else:
                    # If rest is empty, clustering not needed
                    rv = np.zeros((0, 3))
                if rv.size > 0:
                    minv = rv.min(axis=0)
                    maxv = rv.max(axis=0)
                else:
                    minv = np.asarray(orig_o3.vertices).min(axis=0)
                    maxv = np.asarray(orig_o3.vertices).max(axis=0)
                diag = float(np.linalg.norm(maxv - minv))
                # Avoid zero diagonal
                diag = max(diag, 1e-8)
                voxel_size = diag / max(float(target_rest), 1.0) ** (1.0/3.0)
                voxel_size *= float(self.vertex_clustering_factor)
                try:
                    if len(rest_tris) > 0:
                        clustered = mesh_rest.simplify_vertex_clustering(
                            voxel_size=voxel_size,
                            contraction=o3d.geometry.SimplificationContraction.Average,
                        )
                        clustered.remove_degenerate_triangles()
                        clustered.remove_duplicated_triangles()
                        clustered.remove_duplicated_vertices()
                        clustered.remove_non_manifold_edges()
                    else:
                        clustered = mesh_rest
                except Exception as e:
                    self.report({"ERROR"}, f"Vertex clustering failed: {e}")
                    bpy.data.objects.remove(original_obj, do_unlink=True)
                    return {"CANCELLED"}
                # Project clustered vertices onto original mesh
                if len(clustered.vertices) > 0:
                    pts = np.asarray(clustered.vertices, dtype=np.float64)
                    if bool(getattr(self, 'use_directional_projection', False)):
                        # Compute normals for clustered mesh before projection
                        clustered.compute_vertex_normals()
                        norms = np.asarray(clustered.vertex_normals, dtype=np.float64)
                        proj = _project_points_directional(orig_o3, pts, norms)
                        clustered.vertices = o3d.utility.Vector3dVector(proj)
                    else:
                        closest = _closest_points_raycast(orig_o3, pts)
                        clustered.vertices = o3d.utility.Vector3dVector(closest)
                    clustered.compute_vertex_normals()
                # Combine protected and clustered rest meshes
                mesh_combined = mesh_prot + clustered
                mesh_combined.remove_duplicated_vertices()
                mesh_combined.remove_duplicated_triangles()
                mesh_combined.remove_non_manifold_edges()
                mesh_combined.compute_vertex_normals()
                # Quantize
                try:
                    quantize_vertices_o3d(mesh_combined, self.fixed_point_precision_bits)
                except Exception as e:
                    self.report({"WARNING"}, f"Quantization skipped: {e}")
                # Write to Blender
                new_mesh = bpy.data.meshes.new(name="PS1_Decimated")
                open3d_to_mesh(mesh_combined, new_mesh)
                obj.data = new_mesh
                obj.data.update()
                # Optional QuadriFlow
                if self.use_quadriflow_remesh:
                    try:
                        bpy.ops.object.mode_set(mode='OBJECT')
                    except Exception:
                        pass
                    try:
                        bpy.ops.object.quadriflow_remesh(
                            target_faces=max(50, target_total),
                            use_preserve_sharp=True,
                            use_mesh_symmetry=False,
                            use_preserve_boundary=True,
                        )
                        obj.data.update()
                    except Exception as e:
                        self.report({"WARNING"}, f"QuadriFlow failed: {e}")
                # Downsample textures
                for mat in obj.data.materials:
                    if mat is not None:
                        try:
                            replace_material_with_downsampled(
                                mat,
                                tex_size=self.tex_size,
                                export_path=self.texture_export_path or None,
                            )
                        except Exception as e:
                            self.report({"WARNING"}, f"Texture downsampling error: {e}")
                # Displaced subdivision: project subdivided mesh on original surface
                try:
                    base_o3 = mesh_to_open3d(obj)
                    disp_o3 = generate_displaced_subdivision_open3d(
                        base=base_o3,
                        reference=orig_o3,
                        levels=1,
                        apply_displacement=True,
                        directional_projection=bool(getattr(self, 'use_directional_projection', False)),
                    )
                    disp_mesh = bpy.data.meshes.new(name=f"{obj.name}_displaced_mesh")
                    open3d_to_mesh(disp_o3, disp_mesh)
                    disp_obj = bpy.data.objects.new(name=f"{obj.name}_displaced", object_data=disp_mesh)
                    disp_obj.matrix_world = obj.matrix_world.copy()
                    disp_obj.data.materials.clear()
                    for mat in obj.data.materials:
                        disp_obj.data.materials.append(mat)
                    context.scene.collection.objects.link(disp_obj)
                except Exception as e:
                    self.report({"WARNING"}, f"Displaced subdivision generation failed: {e}")
                # Clean up
                bpy.data.objects.remove(original_obj, do_unlink=True)
                self.report({"INFO"}, "PS1 decimation completed with vertex clustering and feature preservation")
                return {"FINISHED"}
            else:
                # No protected vertices; cluster entire mesh
                verts_np = np.asarray(orig_o3.vertices)
                tris_np = np.asarray(orig_o3.triangles)
                # Compute voxel size from bounding box
                if verts_np.size > 0:
                    minv = verts_np.min(axis=0)
                    maxv = verts_np.max(axis=0)
                else:
                    minv = np.zeros(3)
                    maxv = np.ones(3)
                diag = float(np.linalg.norm(maxv - minv))
                diag = max(diag, 1e-8)
                # Use entire target face count for clustering
                voxel_size = diag / max(float(self.poly_count_target), 1.0) ** (1.0/3.0)
                voxel_size *= float(self.vertex_clustering_factor)
                try:
                    clustered = orig_o3.simplify_vertex_clustering(
                        voxel_size=voxel_size,
                        contraction=o3d.geometry.SimplificationContraction.Average,
                    )
                    clustered.remove_degenerate_triangles()
                    clustered.remove_duplicated_triangles()
                    clustered.remove_duplicated_vertices()
                    clustered.remove_non_manifold_edges()
                except Exception as e:
                    self.report({"ERROR"}, f"Vertex clustering failed: {e}")
                    bpy.data.objects.remove(original_obj, do_unlink=True)
                    return {"CANCELLED"}
                # Project vertices back to original surface
                if len(clustered.vertices) > 0:
                    pts = np.asarray(clustered.vertices, dtype=np.float64)
                    if bool(getattr(self, 'use_directional_projection', False)):
                        clustered.compute_vertex_normals()
                        norms = np.asarray(clustered.vertex_normals, dtype=np.float64)
                        proj = _project_points_directional(orig_o3, pts, norms)
                        clustered.vertices = o3d.utility.Vector3dVector(proj)
                    else:
                        closest = _closest_points_raycast(orig_o3, pts)
                        clustered.vertices = o3d.utility.Vector3dVector(closest)
                    clustered.compute_vertex_normals()
                # Quantize
                try:
                    quantize_vertices_o3d(clustered, self.fixed_point_precision_bits)
                except Exception as e:
                    self.report({"WARNING"}, f"Quantization skipped: {e}")
                # Write mesh
                new_mesh = bpy.data.meshes.new(name="PS1_Decimated")
                open3d_to_mesh(clustered, new_mesh)
                obj.data = new_mesh
                obj.data.update()
                # Optional QuadriFlow
                if self.use_quadriflow_remesh:
                    try:
                        bpy.ops.object.mode_set(mode='OBJECT')
                    except Exception:
                        pass
                    try:
                        bpy.ops.object.quadriflow_remesh(
                            target_faces=max(50, int(self.poly_count_target)),
                            use_preserve_sharp=True,
                            use_mesh_symmetry=False,
                            use_preserve_boundary=True,
                        )
                        obj.data.update()
                    except Exception as e:
                        self.report({"WARNING"}, f"QuadriFlow failed: {e}")
                # Downsample textures
                for mat in obj.data.materials:
                    if mat is not None:
                        try:
                            replace_material_with_downsampled(
                                mat,
                                tex_size=self.tex_size,
                                export_path=self.texture_export_path or None,
                            )
                        except Exception as e:
                            self.report({"WARNING"}, f"Texture downsampling error: {e}")
                # Displaced subdivision generation
                try:
                    base_o3 = mesh_to_open3d(obj)
                    disp_o3 = generate_displaced_subdivision_open3d(
                        base=base_o3,
                        reference=orig_o3,
                        levels=1,
                        apply_displacement=True,
                        directional_projection=bool(getattr(self, 'use_directional_projection', False)),
                    )
                    disp_mesh = bpy.data.meshes.new(name=f"{obj.name}_displaced_mesh")
                    open3d_to_mesh(disp_o3, disp_mesh)
                    disp_obj = bpy.data.objects.new(name=f"{obj.name}_displaced", object_data=disp_mesh)
                    disp_obj.matrix_world = obj.matrix_world.copy()
                    disp_obj.data.materials.clear()
                    for mat in obj.data.materials:
                        disp_obj.data.materials.append(mat)
                    context.scene.collection.objects.link(disp_obj)
                except Exception as e:
                    self.report({"WARNING"}, f"Displaced subdivision generation failed: {e}")
                # Clean up
                bpy.data.objects.remove(original_obj, do_unlink=True)
                self.report({"INFO"}, "PS1 decimation completed with vertex clustering")
                return {"FINISHED"}
        if _HAS_O3D and use_protected:
            try:
                # Convert original mesh to Open3D for partitioning
                orig_o3 = mesh_to_open3d(original_obj)
            except Exception as e:
                self.report({"ERROR"}, f"Failed to convert mesh to Open3D: {e}")
                bpy.data.objects.remove(original_obj, do_unlink=True)
                return {"CANCELLED"}
            verts_np = np.asarray(orig_o3.vertices)
            tris_np = np.asarray(orig_o3.triangles)
            # If no vertices marked protected, treat as unprotected
            if is_protected is None or not is_protected.any():
                # fall through to standard decimation below
                use_protected = False
            else:
                # Partition faces into protected and rest based on vertex mask
                protected_face_mask = np.any(is_protected[tris_np], axis=1)
                protected_tris = tris_np[protected_face_mask]
                rest_tris = tris_np[~protected_face_mask]
                # Build protected submesh
                prot_vert_ids = np.unique(protected_tris.flatten())
                prot_vert_map = {int(old_idx): idx for idx, old_idx in enumerate(prot_vert_ids)}
                prot_vertices = verts_np[prot_vert_ids]
                prot_faces_local = np.array([[prot_vert_map[int(v)] for v in face] for face in protected_tris], dtype=np.int32)
                mesh_prot = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(prot_vertices),
                    o3d.utility.Vector3iVector(prot_faces_local),
                )
                # Build rest submesh
                if len(rest_tris) > 0:
                    rest_vert_ids = np.unique(rest_tris.flatten())
                    rest_vert_map = {int(old_idx): idx for idx, old_idx in enumerate(rest_vert_ids)}
                    rest_vertices = verts_np[rest_vert_ids]
                    rest_faces_local = np.array([[rest_vert_map[int(v)] for v in face] for face in rest_tris], dtype=np.int32)
                    mesh_rest = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(rest_vertices),
                        o3d.utility.Vector3iVector(rest_faces_local),
                    )
                else:
                    # No faces to decimate
                    mesh_rest = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(),
                        o3d.utility.Vector3iVector(),
                    )
                # Compute target faces for the rest region
                protected_count = len(protected_tris)
                target_total = int(self.poly_count_target)
                target_rest = max(10, target_total - protected_count)
                # Decimate rest region if there are faces
                try:
                    if len(rest_tris) > 0 and target_rest > 0:
                        mesh_rest_dec = mesh_rest.simplify_quadric_decimation(
                            target_number_of_triangles=target_rest
                        )
                        mesh_rest_dec.remove_degenerate_triangles()
                        mesh_rest_dec.remove_duplicated_triangles()
                        mesh_rest_dec.remove_duplicated_vertices()
                        mesh_rest_dec.remove_non_manifold_edges()
                    else:
                        mesh_rest_dec = mesh_rest
                except Exception as e:
                    self.report({"ERROR"}, f"Open3D protected decimation failed: {e}")
                    bpy.data.objects.remove(original_obj, do_unlink=True)
                    return {"CANCELLED"}
                # Combine protected and decimated rest meshes
                mesh_combined = mesh_prot + mesh_rest_dec
                mesh_combined.remove_duplicated_vertices()
                mesh_combined.remove_duplicated_triangles()
                mesh_combined.remove_non_manifold_edges()
                mesh_combined.compute_vertex_normals()
                # Quantize vertices
                try:
                    quantize_vertices_o3d(mesh_combined, self.fixed_point_precision_bits)
                except Exception as e:
                    self.report({"WARNING"}, f"Quantization skipped: {e}")
                # Write mesh back to Blender
                new_mesh = bpy.data.meshes.new(name="PS1_Decimated")
                open3d_to_mesh(mesh_combined, new_mesh)
                obj.data = new_mesh
                obj.data.update()
                # Optional QuadriFlow remeshing
                if self.use_quadriflow_remesh:
                    try:
                        bpy.ops.object.mode_set(mode='OBJECT')
                    except Exception:
                        pass
                    try:
                        bpy.ops.object.quadriflow_remesh(
                            target_faces=max(50, target_total),
                            use_preserve_sharp=True,
                            use_mesh_symmetry=False,
                            use_preserve_boundary=True,
                        )
                        obj.data.update()
                    except Exception as e:
                        self.report({"WARNING"}, f"QuadriFlow failed: {e}")
                # Downsample textures
                for mat in obj.data.materials:
                    if mat is not None:
                        try:
                            replace_material_with_downsampled(
                                mat,
                                tex_size=self.tex_size,
                                export_path=self.texture_export_path or None,
                            )
                        except Exception as e:
                            self.report({"WARNING"}, f"Texture downsampling error: {e}")
                # Generate displaced subdivision
                try:
                    base_o3 = mesh_to_open3d(obj)
                    disp_o3 = generate_displaced_subdivision_open3d(
                        base=base_o3,
                        reference=orig_o3,
                        levels=1,
                        apply_displacement=True,
                        directional_projection=bool(getattr(self, 'use_directional_projection', False)),
                    )
                    disp_mesh = bpy.data.meshes.new(name=f"{obj.name}_displaced_mesh")
                    open3d_to_mesh(disp_o3, disp_mesh)
                    disp_obj = bpy.data.objects.new(name=f"{obj.name}_displaced", object_data=disp_mesh)
                    disp_obj.matrix_world = obj.matrix_world.copy()
                    # Transfer materials to displaced mesh
                    disp_obj.data.materials.clear()
                    for mat in obj.data.materials:
                        disp_obj.data.materials.append(mat)
                    context.scene.collection.objects.link(disp_obj)
                except Exception as e:
                    self.report({"WARNING"}, f"Displaced subdivision generation failed: {e}")
                # Clean up
                bpy.data.objects.remove(original_obj, do_unlink=True)
                self.report({"INFO"}, "PS1 decimation completed with feature preservation")
                return {"FINISHED"}

        # Case 2: Standard Open3D decimation (no protected vertices or no threshold)
        if _HAS_O3D:
            try:
                orig_o3 = mesh_to_open3d(original_obj)
            except Exception as e:
                self.report({"ERROR"}, f"Failed to convert mesh to Open3D: {e}")
                bpy.data.objects.remove(original_obj, do_unlink=True)
                return {"CANCELLED"}
            try:
                dec_o3 = orig_o3.simplify_quadric_decimation(
                    target_number_of_triangles=max(10, int(self.poly_count_target))
                )
                dec_o3.remove_degenerate_triangles()
                dec_o3.remove_duplicated_triangles()
                dec_o3.remove_duplicated_vertices()
                dec_o3.remove_non_manifold_edges()
                dec_o3.compute_vertex_normals()
            except Exception as e:
                self.report({"ERROR"}, f"Open3D decimation failed: {e}")
                bpy.data.objects.remove(original_obj, do_unlink=True)
                return {"CANCELLED"}
            # Quantize
            try:
                quantize_vertices_o3d(dec_o3, self.fixed_point_precision_bits)
            except Exception as e:
                self.report({"WARNING"}, f"Quantization skipped: {e}")
            # Write to Blender
            new_mesh = bpy.data.meshes.new(name="PS1_Decimated")
            open3d_to_mesh(dec_o3, new_mesh)
            obj.data = new_mesh
            obj.data.update()
            # Optional QuadriFlow
            if self.use_quadriflow_remesh:
                try:
                    bpy.ops.object.mode_set(mode='OBJECT')
                except Exception:
                    pass
                try:
                    bpy.ops.object.quadriflow_remesh(
                        target_faces=max(50, int(self.poly_count_target)),
                        use_preserve_sharp=True,
                        use_mesh_symmetry=False,
                        use_preserve_boundary=True,
                    )
                    obj.data.update()
                except Exception as e:
                    self.report({"WARNING"}, f"QuadriFlow failed: {e}")
            # Downsample textures
            for mat in obj.data.materials:
                if mat is not None:
                    try:
                        replace_material_with_downsampled(
                            mat,
                            tex_size=self.tex_size,
                            export_path=self.texture_export_path or None,
                        )
                    except Exception as e:
                        self.report({"WARNING"}, f"Texture downsampling error: {e}")
            # Displaced subdivision
            try:
                base_o3 = mesh_to_open3d(obj)
                disp_o3 = generate_displaced_subdivision_open3d(
                    base=base_o3,
                    reference=orig_o3,
                    levels=1,
                    apply_displacement=True,
                    directional_projection=bool(getattr(self, 'use_directional_projection', False)),
                )
                disp_mesh = bpy.data.meshes.new(name=f"{obj.name}_displaced_mesh")
                open3d_to_mesh(disp_o3, disp_mesh)
                disp_obj = bpy.data.objects.new(name=f"{obj.name}_displaced", object_data=disp_mesh)
                disp_obj.matrix_world = obj.matrix_world.copy()
                disp_obj.data.materials.clear()
                for mat in obj.data.materials:
                    disp_obj.data.materials.append(mat)
                context.scene.collection.objects.link(disp_obj)
            except Exception as e:
                self.report({"WARNING"}, f"Displaced subdivision generation failed: {e}")
            bpy.data.objects.remove(original_obj, do_unlink=True)
            self.report({"INFO"}, "PS1 decimation completed with Open3D")
            return {"FINISHED"}

        # Case 3: Open3D not available – fallback to Blender decimate modifier
        self.report({"INFO"}, "Open3D unavailable; using Blender decimate modifier")
        # Build ratio for decimate modifier relative to original face count
        orig_face_count = max(1, len(obj.data.polygons))
        ratio = float(self.poly_count_target) / float(orig_face_count)
        ratio = max(0.0, min(1.0, ratio))
        # Duplicate mesh for decimation
        dec_mesh = original_mesh.copy()
        dec_obj = bpy.data.objects.new(name="__ps1_decimate_tmp__", object_data=dec_mesh)
        dec_obj.matrix_world = obj.matrix_world.copy()
        context.scene.collection.objects.link(dec_obj)
        mod = dec_obj.modifiers.new(name="Decimate", type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        # If a group is specified or thresholds triggered automatic protection,
        # assign vertex group to decimate modifier to preserve those vertices.
        if vg_name:
            mod.vertex_group = vg_name
            mod.invert_vertex_group = False
            try:
                mod.vertex_group_factor = 1.0
            except Exception:
                pass
        bpy.context.view_layer.objects.active = dec_obj
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception as e:
            self.report({"ERROR"}, f"Blender decimate modifier failed: {e}")
            bpy.data.objects.remove(dec_obj, do_unlink=True)
            bpy.data.objects.remove(original_obj, do_unlink=True)
            return {"CANCELLED"}
        # Assign decimated mesh to active object
        obj.data = dec_obj.data.copy()
        bpy.data.objects.remove(dec_obj, do_unlink=True)
        # Remove reference object
        bpy.data.objects.remove(original_obj, do_unlink=True)
        self.report({"INFO"}, "PS1 decimation completed with Blender decimate")
        return {"FINISHED"}

        # No protected group or Open3D not available: use standard Open3D decimation
        if _HAS_O3D:
            try:
                orig_o3 = mesh_to_open3d(original_obj)
            except Exception as e:
                self.report({"ERROR"}, f"Failed to convert mesh to Open3D: {e}")
                bpy.data.objects.remove(original_obj, do_unlink=True)
                return {"CANCELLED"}
            try:
                dec_o3 = orig_o3.simplify_quadric_decimation(
                    target_number_of_triangles=max(10, int(self.poly_count_target))
                )
                dec_o3.remove_degenerate_triangles()
                dec_o3.remove_duplicated_triangles()
                dec_o3.remove_duplicated_vertices()
                dec_o3.remove_non_manifold_edges()
                dec_o3.compute_vertex_normals()
            except Exception as e:
                self.report({"ERROR"}, f"Open3D decimation failed: {e}")
                bpy.data.objects.remove(original_obj, do_unlink=True)
                return {"CANCELLED"}
            # Quantize vertices
            try:
                quantize_vertices_o3d(dec_o3, self.fixed_point_precision_bits)
            except Exception as e:
                self.report({"WARNING"}, f"Quantization skipped: {e}")
            # Write decimated mesh back
            new_mesh = bpy.data.meshes.new(name="PS1_Decimated")
            open3d_to_mesh(dec_o3, new_mesh)
            obj.data = new_mesh
            obj.data.update()
            # Optional QuadriFlow
            if self.use_quadriflow_remesh:
                try:
                    bpy.ops.object.mode_set(mode='OBJECT')
                except Exception:
                    pass
                try:
                    bpy.ops.object.quadriflow_remesh(
                        target_faces=max(50, int(self.poly_count_target)),
                        use_preserve_sharp=True,
                        use_mesh_symmetry=False,
                        use_preserve_boundary=True,
                    )
                    obj.data.update()
                except Exception as e:
                    self.report({"WARNING"}, f"QuadriFlow failed: {e}")
            # Downsample textures
            for mat in obj.data.materials:
                if mat is not None:
                    try:
                        replace_material_with_downsampled(
                            mat,
                            tex_size=self.tex_size,
                            export_path=self.texture_export_path or None,
                        )
                    except Exception as e:
                        self.report({"WARNING"}, f"Texture downsampling error: {e}")
            # Generate displaced subdivision
            try:
                base_o3 = mesh_to_open3d(obj)
                disp_o3 = generate_displaced_subdivision_open3d(
                    base=base_o3,
                    reference=orig_o3,
                    levels=1,
                    apply_displacement=True,
                    directional_projection=bool(getattr(self, 'use_directional_projection', False)),
                )
                disp_mesh = bpy.data.meshes.new(name=f"{obj.name}_displaced_mesh")
                open3d_to_mesh(disp_o3, disp_mesh)
                disp_obj = bpy.data.objects.new(name=f"{obj.name}_displaced", object_data=disp_mesh)
                disp_obj.matrix_world = obj.matrix_world.copy()
                disp_obj.data.materials.clear()
                for mat in obj.data.materials:
                    disp_obj.data.materials.append(mat)
                context.scene.collection.objects.link(disp_obj)
            except Exception as e:
                self.report({"WARNING"}, f"Displaced subdivision generation failed: {e}")
            # Clean up
            bpy.data.objects.remove(original_obj, do_unlink=True)
            self.report({"INFO"}, "PS1 decimation completed with Open3D")
            return {"FINISHED"}

        # Fallback: Open3D unavailable; use Blender decimate
        self.report({"INFO"}, "Open3D unavailable; using Blender decimate modifier")
        # Compute ratio relative to original face count
        face_count = max(1, len(obj.data.polygons))
        ratio = float(self.poly_count_target) / float(face_count)
        ratio = max(0.0, min(1.0, ratio))
        # Duplicate original mesh for decimation
        dec_mesh = original_mesh.copy()
        dec_obj = bpy.data.objects.new(name="__ps1_decimate_tmp__", object_data=dec_mesh)
        dec_obj.matrix_world = obj.matrix_world.copy()
        context.scene.collection.objects.link(dec_obj)
        mod = dec_obj.modifiers.new(name="Decimate", type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        # If protected group specified, assign vertex group to modifier
        if self.protected_vertex_group:
            mod.vertex_group = self.protected_vertex_group
            mod.invert_vertex_group = False
            # A vertex group factor of 1 preserves the group completely
            try:
                mod.vertex_group_factor = 1.0
            except Exception:
                pass
        bpy.context.view_layer.objects.active = dec_obj
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception as e:
            self.report({"ERROR"}, f"Blender decimate modifier failed: {e}")
            bpy.data.objects.remove(dec_obj, do_unlink=True)
            bpy.data.objects.remove(original_obj, do_unlink=True)
            return {"CANCELLED"}
        # Assign decimated mesh
        obj.data = dec_obj.data.copy()
        bpy.data.objects.remove(dec_obj, do_unlink=True)
        # Clean up
        bpy.data.objects.remove(original_obj, do_unlink=True)
        self.report({"INFO"}, "PS1 decimation completed with Blender decimate")
        return {"FINISHED"}

        # Fallback: if Open3D unavailable, use Blender's decimate modifier
        # Build decimate ratio relative to original face count
        self.report({"INFO"}, "Open3D unavailable; falling back to Blender decimate modifier")
        # Duplicate object for decimation
        dec_mesh = original_mesh.copy()
        dec_obj = bpy.data.objects.new(name="__ps1_decimate_tmp__", object_data=dec_mesh)
        dec_obj.matrix_world = obj.matrix_world.copy()
        context.scene.collection.objects.link(dec_obj)
        # Compute ratio
        orig_faces = max(1, len(obj.data.polygons))
        ratio = float(self.poly_count_target) / float(orig_faces)
        ratio = max(0.0, min(1.0, ratio))
        mod = dec_obj.modifiers.new(name="Decimate", type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        bpy.context.view_layer.objects.active = dec_obj
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception as e:
            self.report({"ERROR"}, f"Blender decimate modifier failed: {e}")
            bpy.data.objects.remove(dec_obj, do_unlink=True)
            bpy.data.objects.remove(original_obj, do_unlink=True)
            return {"CANCELLED"}
        # Assign decimated mesh to active object
        obj.data = dec_obj.data.copy()
        bpy.data.objects.remove(dec_obj, do_unlink=True)
        # Remove reference object
        bpy.data.objects.remove(original_obj, do_unlink=True)
        self.report({"INFO"}, "PS1 decimation completed with Blender decimate")
        return {"FINISHED"}


def menu_func(self, context: bpy.types.Context) -> None:
    """Add the PS1 decimator to the Object menu."""
    self.layout.operator(OBJECT_OT_ps1_decimate.bl_idname, text="PS1 Decimator")


def register() -> None:
    bpy.utils.register_class(OBJECT_OT_ps1_decimate)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister() -> None:
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    bpy.utils.unregister_class(OBJECT_OT_ps1_decimate)