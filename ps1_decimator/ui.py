import bpy

class OBJECT_PT_ps1_decimate_panel(bpy.types.Panel):

    bl_label = "PS1 Decimator"
    bl_idname = "OBJECT_PT_ps1_decimate_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"
    
    def draw_header(self, context):
        layout = self.layout
        layout.label(text="", icon='MESH_DATA')
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        if not obj or obj.type != 'MESH':
            layout.label(text="No mesh selected", icon='ERROR')
            return
        
        mesh = obj.data
        face_count = len(mesh.polygons)
        vert_count = len(mesh.vertices)
        
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Current Mesh:", icon='INFO')
        col.label(text=f"Vertices: {vert_count:,}")
        col.label(text=f"Faces: {face_count:,}")
        
        wm = context.window_manager
        is_running = context.scene.is_decimating
        
        if is_running:
            layout.separator()
            box = layout.box()
            box.label(text="Decimation in Progress...", icon='TIME')
            box.label(text="Press ESC to cancel")
            box.operator("wm.redraw_timer", text="Refresh", icon='FILE_REFRESH')
        else:
            layout.separator()
            layout.operator("object.ps1_decimate", text="Apply PS1 Decimation", icon='MOD_DECIM')
        
        layout.separator()
        layout.label(text="Decimation Settings:", icon='SETTINGS')
        
        scene = context.scene
        col = layout.column(align=True)
        col.prop(scene, "poly_count_target", slider=True)
        col.prop(scene, "saliency_factor", slider=True)
        col.prop(scene, "angle_factor", slider=True)
        
        layout.separator()
        layout.label(text="PS1 Style Settings:", icon='SHADING_RENDERED')
        
        col = layout.column(align=True)
        col.prop(scene, "fixed_point_precision_bits", slider=True)
        col.prop(scene, "tex_size", slider=True)
        col.prop(scene, "batch_size", slider=True)
        
        col = layout.column()
        col.prop(scene, "texture_export_path")
        
        col = layout.column()
        col.prop(scene, "decimation_progress", text="")


def register_properties():
    """Register scene properties for persistent settings"""
    bpy.types.Scene.poly_count_target = bpy.props.IntProperty(
        name="Target Poly Count",
        description="Number of polygons to keep after decimation",
        default=1000,
        min=1,
        max=100000,
    )
    
    bpy.types.Scene.fixed_point_precision_bits = bpy.props.IntProperty(
        name="Vertex Precision Bits",
        description="Precision of vertex coordinates (lower = more PS1-like)",
        default=12,
        min=1,
        max=32,
    )
    
    bpy.types.Scene.saliency_factor = bpy.props.FloatProperty(
        name="Saliency Factor",
        description="Weight of saliency in cost calculation",
        default=0.75,
        min=0.0,
        max=1.0,
    )
    
    bpy.types.Scene.angle_factor = bpy.props.FloatProperty(
        name="Angle Factor",
        description="Weight of angles in cost calculation",
        default=0.75,
        min=0.0,
        max=1.0,
    )
    
    bpy.types.Scene.tex_size = bpy.props.IntProperty(
        name="Texture Size",
        description="Target texture size after decimation",
        default=64,
        min=1,
        max=1024,
    )
    
    bpy.types.Scene.texture_export_path = bpy.props.StringProperty(
        name="Export Directory",
        description="Directory to export downsampled textures",
        default="",
        subtype="DIR_PATH",
    )
    
    bpy.types.Scene.batch_size = bpy.props.IntProperty(
        name="Batch Size",
        description="Number of edges collapsed per modal tick",
        default=16,
        min=1,
        max=64,
    )
    
    bpy.types.Scene.is_decimating = bpy.props.BoolProperty(
        name="Is Decimating",
        description="Is decimation in progress",
        default=False
    )
    
    bpy.types.Scene.decimation_progress = bpy.props.FloatProperty(
        name="Decimation Progress",
        subtype="PERCENTAGE",
        default=0.0,
        min=0.0,
        max=100.0,
    )

def unregister_properties():
    props_to_remove = [
        'poly_count_target',
        'fixed_point_precision_bits', 
        'saliency_factor',
        'angle_factor',
        'tex_size',
        'texture_export_path',
        'batch_size',
        'is_decimating',
    ]
    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

def register():
    register_properties()
    bpy.utils.register_class(OBJECT_PT_ps1_decimate_panel)

def unregister():
    unregister_properties()
    bpy.utils.unregister_class(OBJECT_PT_ps1_decimate_panel)