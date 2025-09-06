"""
Minimal UI for the PS1 decimator (single-path version).
"""

from __future__ import annotations

import bpy


class PS1_DECIMATOR_PT_panel(bpy.types.Panel):
    bl_label = "PS1 Decimator"
    bl_idname = "PS1_DECIMATOR_PT_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = context.scene
        # Current triangle count for active mesh
        obj = context.active_object
        tri_count = 0
        if obj is not None and obj.type == 'MESH':
            try:
                obj.data.calc_loop_triangles()
                tri_count = len(obj.data.loop_triangles)
            except Exception:
                tri_count = len(obj.data.polygons)
        layout.label(text=f"Current Faces (tri): {tri_count}")

        col = layout.column(align=True)
        col.prop(scene, 'poly_count_target')
        col.prop(scene, 'fixed_point_precision_bits')
        col.prop(scene, 'use_triangulate')
        col.prop(scene, 'keep_modifiers')
        col.separator()
        op = col.operator('object.ps1_decimate', text='Apply PS1 Decimation', icon='MOD_DECIM')
        op.poly_count_target = scene.poly_count_target
        op.fixed_point_precision_bits = scene.fixed_point_precision_bits
        op.use_triangulate = scene.use_triangulate
        op.keep_modifiers = scene.keep_modifiers


def register() -> None:
    from bpy.props import IntProperty, FloatProperty, BoolProperty
    Scene = bpy.types.Scene
    if not hasattr(Scene, 'poly_count_target'):
        Scene.poly_count_target = IntProperty(name="Poly Count Target", default=1000, min=10, max=200000)
    if not hasattr(Scene, 'fixed_point_precision_bits'):
        Scene.fixed_point_precision_bits = IntProperty(name="Fixed-Point Precision (bits)", default=8, min=0, max=16)
    if not hasattr(Scene, 'use_triangulate'):
        Scene.use_triangulate = BoolProperty(name="Force Triangles", default=True)
    if not hasattr(Scene, 'keep_modifiers'):
        Scene.keep_modifiers = BoolProperty(name="Keep Modifiers", default=False)
    bpy.utils.register_class(PS1_DECIMATOR_PT_panel)


def unregister() -> None:
    bpy.utils.unregister_class(PS1_DECIMATOR_PT_panel)
    Scene = bpy.types.Scene
    for prop in ('poly_count_target', 'fixed_point_precision_bits', 'use_triangulate', 'keep_modifiers'):
        if hasattr(Scene, prop):
            delattr(Scene, prop)
