"""Modal PS1-style decimator using Blender's native modifiers.

Runs in small steps to keep the UI responsive:
- Decimate (collapse) → optional Triangulate → Flat shading
- Quantize vertices to 1/(2^bits) grid → Weld by distance
Optional texture downsample + color quantization at the end.
"""

from __future__ import annotations

import time
from typing import Optional

import bpy
import bmesh
from . import materials as ps1m

_PFX = "[PS1]"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _quantize_mesh_vertices(mesh: bpy.types.Mesh, bits: int) -> None:
    """Snap vertex positions to a fixed grid (1/(2^bits))."""
    if bits <= 0:
        return
    scale = float(1 << bits)
    for v in mesh.vertices:
        co = v.co
        co.x = round(co.x * scale) / scale
        co.y = round(co.y * scale) / scale
        co.z = round(co.z * scale) / scale


def _merge_by_distance(mesh: bpy.types.Mesh, dist: float) -> None:
    """Weld coincident vertices within `dist` using bmesh remove_doubles."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(1e-12, float(dist)))
        bm.to_mesh(mesh)
    finally:
        bm.free()


class _ModeGuard:
    """Context manager to temporarily switch object to OBJECT mode."""
    def __init__(self, obj: bpy.types.Object):
        self.obj = obj
        self._mode = obj.mode if obj is not None else 'OBJECT'

    def __enter__(self):
        if self.obj is None:
            return self
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.obj is None:
            return False
        try:
            bpy.ops.object.mode_set(mode=self._mode)
        except Exception:
            pass
        return False


# -----------------------------------------------------------------------------
# Operator (modal)
# -----------------------------------------------------------------------------


class OBJECT_OT_ps1_decimate(bpy.types.Operator):
    """PS1-style decimator using Blender's modifiers (modal)."""

    bl_idname = "object.ps1_decimate"
    bl_label = "PS1 Decimator"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    # Controls
    poly_count_target: bpy.props.IntProperty(  # type: ignore
        name="Target Faces (tri)", default=5000, min=10, soft_max=200000
    )
    fixed_point_precision_bits: bpy.props.IntProperty(  # type: ignore
        name="PS1 Grid Bits",
        description="Quantize vertex positions to 1/(2^bits) units",
        default=8,
        min=0,
        max=16,
    )
    use_triangulate: bpy.props.BoolProperty(  # type: ignore
        name="Force Triangles",
        description="Triangulate after decimation for consistent PS1 topology",
        default=True,
    )
    keep_modifiers: bpy.props.BoolProperty(  # type: ignore
        name="Keep Modifiers",
        description="Leave Decimate/Triangulate modifiers on the object instead of applying",
        default=False,
    )

    # Texture processing
    process_textures: bpy.props.BoolProperty(  # type: ignore
        name="Process Textures",
        description="Downsample and color-quantize textures on materials",
        default=True,
    )
    texture_target_size: bpy.props.IntProperty(  # type: ignore
        name="Texture Size",
        description="Approximate max dimension (px) to scale textures down to",
        default=128,
        min=8,
        soft_max=1024,
    )
    texture_color_bits: bpy.props.IntProperty(  # type: ignore
        name="Color Bit Depth",
        description="Total RGB bits (e.g., 15 → 5 bits/channel). If ≤8, treated as per-channel bits.",
        default=15,
        min=1,
        max=24,
    )
    texture_export_path: bpy.props.StringProperty(  # type: ignore
        name="Export Folder",
        description="Optional folder to save processed textures as PNGs",
        default="",
        subtype='DIR_PATH',
    )

    # Modal state
    _timer = None
    _i: int = 0
    _steps: list[str] = []
    _ratio: float = 1.0
    _obj_name: Optional[str] = None
    _t0: float = 0.0
    _error: Optional[str] = None

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({"ERROR"}, "Select a mesh object")
            return {"CANCELLED"}

        self._obj_name = obj.name
        self._steps = [
            "preflight",
            "compute_ratio",
            "add_decimate",
            "apply_decimate",
            "triangulate" if self.use_triangulate else "skip_triangulate",
            "flat_shading",
            "quantize",
            "weld",
            "finish",
        ]
        self._i = 0
        self._ratio = 1.0
        self._error = None
        self._t0 = time.time()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        try:
            wm.progress_begin(0, 100)
        except Exception:
            pass
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        if event.type != 'TIMER':
            return {"RUNNING_MODAL"}

        try:
            context.window_manager.progress_update(int(100 * (self._i / max(1, len(self._steps)))))
        except Exception:
            pass

        if self._error is not None:
            self.report({"ERROR"}, f"PS1 decimator failed: {self._error}")
            return self._finish(context, cancelled=True)

        if self._i >= len(self._steps):
            total = time.time() - getattr(self, '_t0', time.time())
            self.report({"INFO"}, f"PS1 decimation completed in {total:.2f}s")
            return self._finish(context)

        step = self._steps[self._i]
        self._i += 1

        try:
            obj = bpy.data.objects.get(self._obj_name or "")
            if obj is None or obj.type != 'MESH':
                self._error = "Active mesh not found"
                return {"RUNNING_MODAL"}

            if step == "preflight":
                with _ModeGuard(obj):
                    context.view_layer.objects.active = obj
                    obj.select_set(True)

            elif step == "compute_ratio":
                me = obj.data
                me.calc_loop_triangles()
                cur_tris = len(me.loop_triangles)
                target = max(10, int(self.poly_count_target))
                ratio = float(target) / max(1.0, float(cur_tris))
                self._ratio = float(min(1.0, max(0.01, ratio)))
                print(f"{_PFX} Triangles: cur={cur_tris} target={target} ratio={self._ratio:.4f}")

            elif step == "add_decimate":
                with _ModeGuard(obj):
                    dec = obj.modifiers.new(name="PS1_Decimate", type='DECIMATE')
                    dec.decimate_type = 'COLLAPSE'
                    dec.ratio = self._ratio
                    try:
                        dec.use_collapse_triangulate = True
                    except Exception:
                        pass

            elif step == "apply_decimate":
                if not self.keep_modifiers:
                    with _ModeGuard(obj):
                        try:
                            bpy.ops.object.modifier_apply(modifier="PS1_Decimate")
                        except Exception as e:
                            self._error = f"Apply Decimate failed: {e}"

            elif step == "triangulate":
                with _ModeGuard(obj):
                    tri = obj.modifiers.new(name="PS1_Triangulate", type='TRIANGULATE')
                    try:
                        tri.quad_method = 'FIXED'
                        tri.ngon_method = 'BEAUTY'
                    except Exception:
                        pass
                    if not self.keep_modifiers:
                        try:
                            bpy.ops.object.modifier_apply(modifier="PS1_Triangulate")
                        except Exception as e:
                            self._error = f"Apply Triangulate failed: {e}"

            elif step == "skip_triangulate":
                pass

            elif step == "flat_shading":
                me = obj.data
                for p in me.polygons:
                    p.use_smooth = False
                try:
                    me.use_auto_smooth = False
                except Exception:
                    pass
                me.update()

            elif step == "quantize":
                bits = int(self.fixed_point_precision_bits)
                if bits > 0:
                    _quantize_mesh_vertices(obj.data, bits)

            elif step == "weld":
                bits = int(self.fixed_point_precision_bits)
                if bits > 0:
                    grid = 1.0 / float(1 << bits)
                    eps = max(1e-6, grid * 0.51)
                    _merge_by_distance(obj.data, eps)
                    obj.data.update()

            elif step == "finish":
                # Optional texture processing on materials (downsample + quantize)
                if bool(getattr(self, 'process_textures', True)):
                    try:
                        ps1m.process_object_materials(
                            obj,
                            tex_size=int(getattr(self, 'texture_target_size', 128)),
                            color_bit_depth=int(getattr(self, 'texture_color_bits', 15)),
                            export_path=(getattr(self, 'texture_export_path', '').strip() or None),
                        )
                        print(f"{_PFX} Textures processed: size={getattr(self,'texture_target_size',128)} color_bits={getattr(self,'texture_color_bits',15)} export={(getattr(self,'texture_export_path','') or 'none')}")
                    except Exception as e:
                        print(f"{_PFX} Texture processing failed: {e}")

        except Exception as e:
            self._error = str(e)

        return {"RUNNING_MODAL"}

    def _finish(self, context: bpy.types.Context, cancelled: bool = False):
        wm = context.window_manager
        if getattr(self, "_timer", None) is not None:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        try:
            context.window_manager.progress_end()
        except Exception:
            pass
        return {"CANCELLED"} if cancelled else {"FINISHED"}


def menu_func(self, context: bpy.types.Context) -> None:
    self.layout.operator(OBJECT_OT_ps1_decimate.bl_idname, text="PS1 Decimator")


def register() -> None:
    bpy.utils.register_class(OBJECT_OT_ps1_decimate)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister() -> None:
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    bpy.utils.unregister_class(OBJECT_OT_ps1_decimate)

