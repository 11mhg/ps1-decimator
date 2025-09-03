"""
Utility functions for PS1 decimator materials.

This module provides a function to replace an object's material textures with
downsampled versions, giving the characteristic pixelated look of PS1 era
graphics.  The implementation here is intentionally simple: it duplicates each
image texture used by a material, scales it down to a specified size and then
assigns the scaled image back to the material.  It sets the interpolation to
"Closest" so that the texture is not smoothed when rendered.  If an export
directory is provided, the downsampled images are saved as PNG files.

In the context of the enhanced decimator, this module is used by
``operators.py`` during cleanup to downsample textures for the final mesh.
"""

from __future__ import annotations

import os
from typing import Optional

import bpy


def replace_material_with_downsampled(
    material: bpy.types.Material,
    tex_size: int,
    export_path: Optional[str] | None = None,
) -> None:
    """
    Replace the image textures of a material with downsampled versions.

    Parameters
    ----------
    material : bpy.types.Material
        The material whose textures will be downsampled.  The material must
        have a node tree (i.e. use nodes).
    tex_size : int
        The approximate size (width and height) to downsample textures to.
        If the original texture is smaller than ``tex_size`` on an axis, it
        will not be upsampled.
    export_path : str, optional
        If provided, downsampled textures will be saved as PNG files in this
        directory.  The directory must exist or be creatable.

    Notes
    -----
    This function iterates over all texture image nodes in the material's
    node tree.  For each image, it creates a copy using
    ``bpy.data.images.new`` with the desired size and copies pixels from the
    original image using Blender's built‑in ``scale`` function.  It sets
    ``image.interpolation = 'Closest'`` to disable filtering.  If
    ``export_path`` is supplied, the downsampled image is saved as a PNG.
    """
    if material is None or not material.use_nodes:
        return
    nt = material.node_tree
    if nt is None:
        return
    # ensure export directory exists if provided
    if export_path:
        try:
            os.makedirs(export_path, exist_ok=True)
        except Exception:
            pass
    for node in nt.nodes:
        if node.type == 'TEX_IMAGE' and getattr(node, 'image', None):
            img = node.image
            # Determine new dimensions: downsample while preserving aspect ratio
            orig_w, orig_h = img.size
            # compute scaling factors; ensure at least 1 pixel in each dimension
            scale_x = max(1, int(orig_w / tex_size))
            scale_y = max(1, int(orig_h / tex_size))
            new_w = max(1, int(orig_w / scale_x))
            new_h = max(1, int(orig_h / scale_y))
            # Create a new image and copy pixels
            new_name = f"{img.name}_ps1_{new_w}x{new_h}"
            new_img = bpy.data.images.new(name=new_name, width=new_w, height=new_h)
            # Copy pixel data by sampling the original image; use Blender's
            # ``scale`` to resize in place
            try:
                # Duplicate the original image so that scaling does not affect it
                temp_copy = img.copy()
                temp_copy.scale(new_w, new_h)
                new_img.pixels = temp_copy.pixels[:]
                # Free the temporary copy
                bpy.data.images.remove(temp_copy)
            except Exception:
                # Fallback: fill with average color
                avg = [0.0, 0.0, 0.0, 1.0]
                try:
                    if len(img.pixels) >= 4:
                        # compute average color of original
                        pixels = list(img.pixels)
                        count = len(pixels) // 4
                        total = [0.0, 0.0, 0.0, 0.0]
                        for i in range(0, len(pixels), 4):
                            for c in range(4):
                                total[c] += pixels[i + c]
                        avg = [c / count for c in total]
                except Exception:
                    pass
                # fill new image with average color
                new_img.pixels = avg * (new_w * new_h)
            # assign new image to node
            node.image = new_img
            # set pixel interpolation to closest to achieve pixelated effect
            try:
                new_img.interpolation = 'Closest'
            except Exception:
                pass
            # export if required
            if export_path:
                try:
                    file_name = f"{new_name}.png"
                    file_path = os.path.join(export_path, file_name)
                    new_img.filepath_raw = file_path
                    new_img.file_format = 'PNG'
                    new_img.save()
                except Exception:
                    pass
    return