"""
PS1-style texture processing helpers.

Downsamples material textures to a target size and quantizes their colors to a
limited bit depth for a retro look. New images are created and set to use
'Closest' interpolation; originals remain untouched.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Tuple

import bpy


def _quantize_rgba_buffer(pixels: list[float], bits_per_channel: int) -> list[float]:
    """Quantize RGB channels of an RGBA buffer to N bits/channel in-place.

    Returns the same list. Alpha is left unchanged.
    """
    chan_bits = max(1, min(8, int(bits_per_channel)))
    if chan_bits >= 8:
        return pixels
    levels = (1 << chan_bits) - 1
    step = 1.0 / float(levels)

    def quantize(v: float) -> float:
        return max(0.0, min(1.0, round(v / step) * step))

    for i in range(0, len(pixels), 4):
        pixels[i + 0] = quantize(pixels[i + 0])
        pixels[i + 1] = quantize(pixels[i + 1])
        pixels[i + 2] = quantize(pixels[i + 2])
    return pixels


def replace_material_with_downsampled(
    material: bpy.types.Material,
    tex_size: int,
    color_bits_per_channel: int = 5,
    export_path: Optional[str] | None = None,
    _cache: Optional[Dict[Tuple[str, int, int], bpy.types.Image]] = None,
) -> None:
    """Downsample and color‑quantize all image textures used by a material.

    Creates a new image for each source, assigns it back to the node, and sets
    interpolation to 'Closest'. If `export_path` is given, saves PNG files there.
    """
    if material is None or not material.use_nodes:
        return
    node_tree = material.node_tree
    if node_tree is None:
        return
    if export_path:
        try:
            os.makedirs(export_path, exist_ok=True)
        except Exception:
            pass
    cache: Dict[Tuple[str, int, int], bpy.types.Image] = _cache or {}
    for node in node_tree.nodes:
        if node.type != 'TEX_IMAGE' or not getattr(node, 'image', None):
            continue
        image = node.image
        orig_w, orig_h = image.size
        scale_x = max(1, int(orig_w / tex_size))
        scale_y = max(1, int(orig_h / tex_size))
        new_w = max(1, int(orig_w / scale_x))
        new_h = max(1, int(orig_h / scale_y))
        key = (image.name, new_w, int(max(1, color_bits_per_channel)))
        if key in cache and cache[key] in bpy.data.images:
            node.image = cache[key]
            try:
                node.image.interpolation = 'Closest'
            except Exception:
                pass
            continue

        new_name = f"{image.name}_ps1_{new_w}x{new_h}"
        new_image = bpy.data.images.new(name=new_name, width=new_w, height=new_h)
        try:
            temp_copy = image.copy()
            temp_copy.scale(new_w, new_h)
            pixels = list(temp_copy.pixels[:])
            _quantize_rgba_buffer(pixels, int(max(1, color_bits_per_channel)))
            new_image.pixels = pixels
            bpy.data.images.remove(temp_copy)
        except Exception:
            avg_color = [0.0, 0.0, 0.0, 1.0]
            try:
                src = list(image.pixels)
                count = max(1, len(src) // 4)
                totals = [0.0, 0.0, 0.0, 0.0]
                for i in range(0, len(src), 4):
                    totals[0] += src[i + 0]
                    totals[1] += src[i + 1]
                    totals[2] += src[i + 2]
                    totals[3] += src[i + 3]
                avg_color = [c / count for c in totals]
            except Exception:
                pass
            quantized_avg = avg_color[:]
            _quantize_rgba_buffer(quantized_avg, int(max(1, color_bits_per_channel)))
            new_image.pixels = quantized_avg * (new_w * new_h)

        node.image = new_image
        cache[key] = new_image
        try:
            new_image.interpolation = 'Closest'
        except Exception:
            pass
        if export_path:
            try:
                file_name = f"{new_name}.png"
                file_path = os.path.join(export_path, file_name)
                new_image.filepath_raw = file_path
                new_image.file_format = 'PNG'
                new_image.save()
            except Exception:
                pass


def process_object_materials(
    obj: bpy.types.Object,
    tex_size: int,
    color_bit_depth: int = 15,
    export_path: Optional[str] | None = None,
) -> None:
    """Process all materials on a mesh object: downsample and quantize textures.

    If `color_bit_depth` > 8, it is treated as total RGB bits (e.g., 15 → 5 per
    channel). Otherwise it is used as bits per channel directly.
    """
    if obj is None or obj.type != 'MESH':
        return
    bits_per_channel = int(color_bit_depth)
    if bits_per_channel > 8:
        bits_per_channel = max(1, min(8, bits_per_channel // 3))
    else:
        bits_per_channel = max(1, min(8, bits_per_channel))
    cache: Dict[Tuple[str, int, int], bpy.types.Image] = {}
    material_list = getattr(obj.data, 'materials', None)
    if not material_list:
        return
    for material in material_list:
        if material is None:
            continue
        replace_material_with_downsampled(
            material=material,
            tex_size=int(max(1, tex_size)),
            color_bits_per_channel=bits_per_channel,
            export_path=export_path,
            _cache=cache,
        )
