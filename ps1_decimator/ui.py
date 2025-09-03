"""
User interface for the PS1 decimator add‑on.

This module defines a panel in Blender's 3D Viewport Sidebar that exposes
properties of the PS1 decimator operator.  Users can set the target poly
count, fixed‑point precision, saliency and parameterization factors, feature
factor (for preserving ridges and furrows), texture size and export path.  A
progress bar displays the decimation progress.  The panel also provides a
button to start the decimation operator.

The UI uses Scene properties as a central store so that values persist
between sessions.  When the operator runs, it reads these scene properties
and applies them.  See ``operators.py`` for details on the decimation
algorithm.
"""

from __future__ import annotations

import bpy


class PS1_DECIMATOR_PT_panel(bpy.types.Panel):
    """UI panel for the PS1 decimator operator.

    This panel lives in the Object Properties editor (Properties → Object tab).
    It exposes the PS1 decimator settings and a button to run the operator.
    """

    bl_label = "PS1 Decimator"
    bl_idname = "PS1_DECIMATOR_PT_panel"
    # Place the panel in the Properties editor under the Object context so
    # users can find it in the object panel.  The region is 'WINDOW' by
    # default for Properties.  Setting bl_context to 'object' ensures the
    # panel appears when the Object Properties tab is selected.
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = context.scene
        col = layout.column(align=True)
        col.prop(scene, 'poly_count_target')
        col.prop(scene, 'fixed_point_precision_bits')
        col.prop(scene, 'saliency_factor')
        col.prop(scene, 'angle_factor')
        col.prop(scene, 'param_factor')
        col.prop(scene, 'feature_factor')
        col.prop(scene, 'tex_size')
        col.prop(scene, 'texture_export_path')
        col.prop(scene, 'protected_vertex_group')
        # batch_size is no longer used by the trimesh operator but remains for
        # backward compatibility; omit it from the UI.
        col.prop(scene, 'use_quadriflow_remesh')
        col.prop(scene, 'use_vertex_clustering_remesh')
        col.prop(scene, 'vertex_clustering_factor')
        col.prop(scene, 'use_directional_projection')
        # Provide explanatory toggle for directional projection.  The panel
        # arranges options logically: directional projection helps avoid
        # snapping vertices to unrelated surfaces when projecting the
        # subdivided mesh back onto the original.
        col.separator()
        # Invoke the decimation operator and copy properties from the scene so
        # the operator uses the values set in the panel.  Without assigning
        # these, the operator would use its default values instead of the
        # persistent scene properties.
        op = col.operator('object.ps1_decimate', text='Apply PS1 Decimation', icon='MOD_DECIM')
        # Transfer scene properties to operator properties.  See operators.py
        op.poly_count_target = scene.poly_count_target
        op.fixed_point_precision_bits = scene.fixed_point_precision_bits
        op.saliency_factor = scene.saliency_factor
        op.angle_factor = scene.angle_factor
        op.param_factor = scene.param_factor
        op.feature_factor = scene.feature_factor
        op.tex_size = int(scene.tex_size)
        op.texture_export_path = scene.texture_export_path
        op.use_quadriflow_remesh = scene.use_quadriflow_remesh
        op.protected_vertex_group = scene.protected_vertex_group
        op.use_vertex_clustering_remesh = scene.use_vertex_clustering_remesh
        op.vertex_clustering_factor = scene.vertex_clustering_factor
        op.use_directional_projection = scene.use_directional_projection
        # The trimesh operator runs to completion immediately, so no progress bar
        # is displayed.


def register() -> None:
    from bpy.props import IntProperty, FloatProperty, StringProperty, BoolProperty
    Scene = bpy.types.Scene
    # register scene properties if they do not already exist
    if not hasattr(Scene, 'poly_count_target'):
        Scene.poly_count_target = IntProperty(
            name="Poly Count Target",
            description="Number of polygons to keep after decimation",
            default=1000,
            min=1,
            max=100000,
        )
    if not hasattr(Scene, 'fixed_point_precision_bits'):
        Scene.fixed_point_precision_bits = IntProperty(
            name="Fixed-Point Precision (bits)",
            description="Precision of vertex coordinates after quantization",
            default=12,
            min=1,
            max=32,
        )
    if not hasattr(Scene, 'saliency_factor'):
        Scene.saliency_factor = FloatProperty(
            name="Saliency Factor",
            description="Weight of curvature (saliency) in the cost function",
            default=0.75,
            min=0.0,
            max=1.0,
        )
    if not hasattr(Scene, 'angle_factor'):
        Scene.angle_factor = FloatProperty(
            name="Angle Factor",
            description="Influence of dihedral angle on collapse cost",
            default=0.75,
            min=0.0,
            max=1.0,
        )
    if not hasattr(Scene, 'param_factor'):
        Scene.param_factor = FloatProperty(
            name="Parameterization Factor",
            description="Weight of UV distortion in cost calculation",
            default=0.25,
            min=0.0,
            max=1.0,
        )
    if not hasattr(Scene, 'feature_factor'):
        Scene.feature_factor = FloatProperty(
            name="Feature Factor",
            description="Weight of normal difference (feature) in cost calculation",
            default=0.25,
            min=0.0,
            max=1.0,
        )
    if not hasattr(Scene, 'tex_size'):
        Scene.tex_size = FloatProperty(
            name="Texture Size",
            description="Approximate size of downsampled textures",
            default=128,
            min=1.0,
            max=1024.0,
        )
    if not hasattr(Scene, 'texture_export_path'):
        Scene.texture_export_path = StringProperty(
            name="Texture Export Path",
            description="Directory to save downsampled textures",
            default="",
            subtype='DIR_PATH',
        )
    if not hasattr(Scene, 'batch_size'):
        Scene.batch_size = IntProperty(
            name="Batch Size",
            description="Number of edges collapsed per modal iteration",
            default=128,
            min=64,
            max=1024,
        )
    if not hasattr(Scene, 'use_quadriflow_remesh'):
        Scene.use_quadriflow_remesh = BoolProperty(
            name="Use QuadriFlow",
            description="Run QuadriFlow remesher on the final mesh during cleanup",
            default=False,
        )

    if not hasattr(Scene, 'use_vertex_clustering_remesh'):
        Scene.use_vertex_clustering_remesh = BoolProperty(
            name="Use Vertex Clustering",
            description=(
                "Simplify mesh via vertex clustering rather than QEM decimation. "
                "Vertices are clustered into a voxel grid and then snapped back to "
                "the original surface."
            ),
            default=False,
        )

    if not hasattr(Scene, 'vertex_clustering_factor'):
        Scene.vertex_clustering_factor = FloatProperty(
            name="Clustering Factor",
            description="Scaling factor for voxel size used during vertex clustering",
            default=1.0,
            min=0.01,
            max=10.0,
        )

    if not hasattr(Scene, 'protected_vertex_group'):
        Scene.protected_vertex_group = StringProperty(
            name="Protected Group",
            description="Name of vertex group whose vertices should be preserved",
            default="",
        )

    if not hasattr(Scene, 'use_directional_projection'):
        Scene.use_directional_projection = BoolProperty(
            name="Directional Projection",
            description=(
                "Project subdivided vertices along their normals when snapping "
                "back to the original surface.  This helps avoid snapping onto "
                "nearby but unrelated surfaces."
            ),
            default=True,
        )
    if not hasattr(Scene, 'is_decimating'):
        Scene.is_decimating = BoolProperty(
            name="Is Decimating",
            description="Indicates whether decimation is in progress",
            default=False,
        )
    if not hasattr(Scene, 'decimation_progress'):
        Scene.decimation_progress = FloatProperty(
            name="Decimation Progress",
            description="Progress of the decimation operation (0–100)",
            default=0.0,
            min=0.0,
            max=100.0,
        )
    bpy.utils.register_class(PS1_DECIMATOR_PT_panel)


def unregister() -> None:
    bpy.utils.unregister_class(PS1_DECIMATOR_PT_panel)
    # Remove custom scene properties only if they exist (to avoid KeyError on reload)
    Scene = bpy.types.Scene
    for prop in (
        'poly_count_target', 'fixed_point_precision_bits', 'saliency_factor',
        'angle_factor', 'param_factor', 'feature_factor', 'tex_size',
        'texture_export_path', 'batch_size', 'use_quadriflow_remesh',
        'use_vertex_clustering_remesh', 'vertex_clustering_factor',
        'protected_vertex_group', 'use_directional_projection',
        'is_decimating', 'decimation_progress'
    ):
        if hasattr(Scene, prop):
            delattr(Scene, prop)