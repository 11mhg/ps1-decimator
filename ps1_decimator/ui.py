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
        layout.operator("object.ps1_decimate", text="Apply Decimation")
        
        layout.prop(context.scene, 'ratio', slider=True)
        layout.prop(context.scene, 'scale_factor', slider=True)
        layout.prop(context.scene, 'uv_steps', slider=True)
        layout.prop(context.scene, 'saliency_factor', slider=True)
        layout.prop(context.scene, 'angle_factor', slider=True)
        layout.prop(context.scene, 'tex_size', slider=True)


def register_properties():
    bpy.types.Scene.ratio = bpy.props.FloatProperty(
        name="Target Ratio", default=0.2, min=0.01, max=1.0
    )
    bpy.types.Scene.scale_factor = bpy.props.FloatProperty(
        name="Scaling Factor for Fixed Precision Grid", default=4096., min=1.0, max=8192.
    )
    bpy.types.Scene.uv_steps = bpy.props.IntProperty(
        name="UV Steps", default=16, min=2, max=512
    )
    bpy.types.Scene.saliency_factor = bpy.props.FloatProperty(
        name="Saliency Factor", default=0.5, min=0., max=1.
    )
    bpy.types.Scene.angle_factor = bpy.props.FloatProperty(
        name="Angle Factor", default=0.0, min=0., max=1.
    )
    bpy.types.Scene.tex_size = bpy.props.IntProperty(
        name="Texture Size", default=128, min=1, max=1024
    )

def unregister_properties():
    del bpy.types.Scene.ratio
    del bpy.types.Scene.scale_factor
    del bpy.types.Scene.uv_steps
    del bpy.types.Scene.saliency_factor

def register():
    register_properties()
    bpy.utils.register_class(OBJECT_PT_ps1_decimate_panel)
    
def unregister():
    unregister_properties()
    bpy.utils.unregister_class(OBJECT_PT_ps1_decimate_panel)