import bpy

class OBJECT_PT_ps1_decimate_panel(bpy.types.Panel):
    
    bl_label = "PS1 Decimator"
    bl_idname = "OBJECT_PT_ps1_decimate_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        layout.label(text="Stylized PS1 Decimation")
        
        if obj.data and len(obj.data.polygons) > 0:
            layout.label(text=f"Current Poly Count: {len(obj.data.polygons):,}")
        
        layout.operator("object.ps1_decimate", text="Apply Decimation")
        
        layout.prop(context.scene, 'poly_count_target', slider=True)
        layout.prop(context.scene, 'fixed_point_precision_bits', slider=True)
        layout.prop(context.scene, 'saliency_factor', slider=True)
        layout.prop(context.scene, 'angle_factor', slider=True)
        layout.prop(context.scene, 'tex_size', slider=True)
        layout.prop(context.scene, 'texture_export_path', )


def register_properties():
    bpy.types.Scene.poly_count_target = bpy.props.IntProperty(
        name="Poly Count Target", default=1000, min=1, max=100000
    )
    bpy.types.Scene.fixed_point_precision_bits = bpy.props.IntProperty(
        name="Fixed-Point Precision in Bits", default=12, min=1, max=32
    )
    bpy.types.Scene.saliency_factor = bpy.props.FloatProperty(
        name="Saliency Factor", default=0.75, min=0., max=1.
    )
    bpy.types.Scene.angle_factor = bpy.props.FloatProperty(
        name="Angle Factor", default=0.75, min=0., max=1.
    )
    bpy.types.Scene.tex_size = bpy.props.IntProperty(
        name="Texture Size", default=128, min=1, max=1024
    )
    bpy.types.Scene.texture_export_path = bpy.props.StringProperty(
        name="Texture Export Path", default="", subtype="DIR_PATH",
    )

def unregister_properties():
    del bpy.types.Scene.poly_count_target
    del bpy.types.Scene.fixed_point_precision_bits
    del bpy.types.Scene.saliency_factor
    del bpy.types.Scene.angle_factor
    del bpy.types.Scene.tex_size
    del bpy.types.Scene.texture_export_path

def register():
    register_properties()
    bpy.utils.register_class(OBJECT_PT_ps1_decimate_panel)
    
def unregister():
    unregister_properties()
    bpy.utils.unregister_class(OBJECT_PT_ps1_decimate_panel)