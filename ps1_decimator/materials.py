import bpy
import numpy as np

def quantize_image(image, num_bits=5):
    if not image.has_data:
        image.pixels[:]
        
    channels = image.channels
    pixels = np.array(image.pixels)
    
    if channels == 1:
        pixels = np.round(pixels * (2**num_bits - 1)) / (2**num_bits - 1)
    else:
        pixels_rgba = pixels.reshape(-1, channels)
        pixels_rgba[:, :3] = np.round(pixels_rgba[:, :3] * (2**num_bits - 1)) / (2**num_bits - 1)
        pixels = pixels_rgba.flatten()
    
    image.pixels = pixels
    image.update()
    return
    

def replace_material_with_downsampled(material, tex_size):
    if not material.node_tree:
        return

    for node in material.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            src_img = node.image

            # Calculate new size while preserving aspect ratio
            aspect = src_img.size[0] / src_img.size[1]
            if aspect >= 1.0:
                new_width = int(tex_size)
                new_height = int(tex_size / aspect)
            else:
                new_width = int(tex_size * aspect)
                new_height = int(tex_size)

            # Duplicate the image
            downsampled = src_img.copy()
            downsampled.name = f"{src_img.name}_downsampled"

            # Rescale in-place (Blender API handles pixel resampling)
            downsampled.scale(new_width, new_height)
            quantize_image(downsampled, num_bits=5)

            node.interpolation = 'Closest'
            node.image = downsampled