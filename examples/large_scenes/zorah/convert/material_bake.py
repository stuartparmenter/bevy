#!/usr/bin/env python3
"""Flatten Zorah's known UE material-layer conventions into Bevy PBR textures.

This is deliberately a Zorah converter, not a general Unreal material compiler.
It resolves material-instance inheritance, material-layer/material-blend defaults,
and the parameter families used by the downloadable UE 5.4 sample.  The output
is a runtime material manifest containing ordinary Base Color, Normal, and ORM
textures that both Bevy meshlet rasterization and Solari can consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


BASE_NAMES = (
    "basecolortexture",
    "diffusetexture",
    "basecolor",
    "albedo",
    "diffuse",
    "color",
    "marblebasecolor",
    "goldbasecolor",
)
NORMAL_NAMES = (
    "normal",
    "normalmap",
    "normaltexture",
    "marblenormal",
    "marblechippingnormal",
    "goldbasenormal",
)
SURFACE_NAMES = (
    "orm",
    "ors",
    "occlusionroughnessmetallic",
    "packedorm",
    "marbleorm",
    "goldbaseorm",
    "rot",
)
EMISSIVE_NAMES = ("emissive", "emissivemask", "emissivetexture", "emission")
EMISSIVE_TEXTURE_NAMES = (*EMISSIVE_NAMES, "extra")
EMISSIVE_INTENSITY_NAMES = (
    "Global Emission Intensity",
    "Emissive Intensity",
    "EmissiveIntensity",
    "Emission Intensity",
)
EMISSIVE_COLOR_NAMES = ("Emissive Color", "Emission Color", "Emission Tint")
HEX_COLOR = re.compile(r"^([0-9a-fA-F]{6}|[0-9a-fA-F]{8})")
GLOBAL = "GlobalParameter"
LAYER = "LayerParameter"
BLEND = "BlendParameter"
MATERIAL_BAKE_PIPELINE_VERSION = 5
LAYER_MATERIAL_BAKE_PIPELINE_VERSION = 5
# The runtime binds textures by parameter name, so every selected texture is
# emitted under the name main.rs looks for rather than the authoring name.
RUNTIME_TEXTURE_NAMES = {
    "base": "Base Color",
    "normal": "Normal",
    "surface": "ORM",
    "emissive": "Emissive",
}
PASSTHROUGH_LAYER = "/Game/Materials/ParentMaterialLayers/LS_Layer_Passthrough.LS_Layer_Passthrough"


def normalized(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parameter_key(parameter: dict[str, Any]) -> tuple[str, str, int]:
    return (
        normalized(str(parameter.get("name", ""))),
        str(parameter.get("association") or GLOBAL),
        int(parameter.get("index", -1)),
    )


@dataclass
class EffectiveMaterial:
    object: str
    package: str
    type: str
    parent: str | None = None
    layers: list[str] = field(default_factory=list)
    blends: list[str] = field(default_factory=list)
    scalars: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict)
    vectors: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict)
    textures: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict)
    switches: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict)
    base_overrides: dict[str, Any] = field(default_factory=dict)

    def clone(self, *, object_name: str | None = None) -> "EffectiveMaterial":
        return EffectiveMaterial(
            object=object_name or self.object,
            package=self.package,
            type=self.type,
            parent=self.parent,
            layers=list(self.layers),
            blends=list(self.blends),
            scalars=dict(self.scalars),
            vectors=dict(self.vectors),
            textures=dict(self.textures),
            switches=dict(self.switches),
            base_overrides=dict(self.base_overrides),
        )

    def merge_record(self, record: dict[str, Any]) -> None:
        for field_name, target in (
            ("scalars", self.scalars),
            ("vectors", self.vectors),
            ("textures", self.textures),
            ("static_switches", self.switches),
        ):
            for parameter in record.get(field_name, []):
                target[parameter_key(parameter)] = parameter
        self.base_overrides.update(record.get("base_overrides", {}))

    def merge_scoped(self, source: "EffectiveMaterial", association: str, index: int) -> None:
        for source_map, target in (
            (source.scalars, self.scalars),
            (source.vectors, self.vectors),
            (source.textures, self.textures),
            (source.switches, self.switches),
        ):
            for parameter in source_map.values():
                scoped = dict(parameter)
                scoped["association"] = association
                scoped["index"] = index
                target[parameter_key(scoped)] = scoped

    def select(
        self,
        field_name: str,
        names: Iterable[str],
        association: str = GLOBAL,
        index: int = -1,
    ) -> dict[str, Any] | None:
        values = getattr(self, field_name)
        for name in names:
            parameter = values.get((normalized(name), association, index))
            if parameter is not None and parameter.get("value") is not None:
                return parameter
        return None

    def scalar(
        self,
        names: Iterable[str],
        association: str = GLOBAL,
        index: int = -1,
        default: float = 0.0,
    ) -> float:
        parameter = self.select("scalars", names, association, index)
        try:
            value = float(parameter["value"]) if parameter is not None else default
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def switch(
        self,
        names: Iterable[str],
        association: str = GLOBAL,
        index: int = -1,
        default: bool = False,
    ) -> bool:
        parameter = self.select("switches", names, association, index)
        return bool(parameter["value"]) if parameter is not None else default


class Resolver:
    def __init__(self, manifest: dict[str, Any]):
        self.records = {record["object"]: record for record in manifest["materials"]}
        self.cache: dict[str, EffectiveMaterial] = {}

    def resolve(self, object_name: str, stack: tuple[str, ...] = ()) -> EffectiveMaterial:
        if object_name in self.cache:
            return self.cache[object_name].clone()
        if object_name in stack:
            raise ValueError(f"material inheritance cycle: {' -> '.join((*stack, object_name))}")
        record = self.records.get(object_name)
        if record is None:
            raise KeyError(f"material record is missing: {object_name}")
        parent = record.get("parent")
        if parent and parent in self.records:
            result = self.resolve(parent, (*stack, object_name))
            result.object = object_name
            result.package = record["package"]
            result.type = record["type"]
        else:
            result = EffectiveMaterial(object_name, record["package"], record["type"])
        result.parent = parent
        result.layers = list(record.get("layers", []))
        result.blends = list(record.get("blends", []))
        for index, layer in enumerate(result.layers):
            if layer in self.records:
                result.merge_scoped(self.resolve(layer, (*stack, object_name)), LAYER, index)
        for index, blend in enumerate(result.blends):
            if blend in self.records:
                result.merge_scoped(self.resolve(blend, (*stack, object_name)), BLEND, index)
        result.merge_record(record)
        self.cache[object_name] = result.clone()
        return result


def parse_color(value: Any, default: tuple[float, float, float, float]) -> np.ndarray:
    if isinstance(value, str):
        match = HEX_COLOR.match(value.strip())
        if match:
            text = match.group(1)
            if len(text) == 6:
                text += "FF"
            components = np.asarray(
                [int(text[offset : offset + 2], 16) / 255.0 for offset in range(0, 8, 2)],
                dtype=np.float32,
            )
            # CUE4Parse prints FLinearColor through FColor, so the hex carries
            # sRGB-encoded RGB with linear alpha. main.rs decodes it the same
            # way (Color::srgba_u8); everything here works in linear.
            components[:3] = srgb_to_linear(components[:3])
            return components
    if isinstance(value, list) and len(value) >= 3:
        components = [float(component) for component in value[:4]]
        if len(components) == 3:
            components.append(1.0)
        return np.asarray(components, dtype=np.float32)
    if isinstance(value, dict):
        components = [value.get(key) for key in ("R", "G", "B", "A")]
        if all(component is not None for component in components[:3]):
            return np.asarray(
                [float(component) if component is not None else 1.0 for component in components],
                dtype=np.float32,
            )
    return np.asarray(default, dtype=np.float32)


def selected_color(
    material: EffectiveMaterial,
    names: Iterable[str],
    association: str,
    index: int,
    default: tuple[float, float, float, float],
) -> np.ndarray:
    parameter = material.select("vectors", names, association, index)
    return parse_color(parameter.get("value") if parameter else None, default)


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1.0 / 2.4) - 0.055)


def hex_color(color: np.ndarray) -> str:
    """Quantize a linear color into the FColor hex parse_color reads back."""
    encoded = linear_to_srgb(np.asarray(color, dtype=np.float32)[:3])
    alpha = float(np.clip(color[3], 0.0, 1.0)) if len(color) > 3 else 1.0
    return "".join(
        f"{round(float(component) * 255):02X}" for component in (*encoded, alpha)
    )


def out_of_range_components(color: np.ndarray) -> bool:
    """Report colors hex_color has to clamp, which cannot round-trip."""
    values = np.asarray(color, dtype=np.float32)[:4]
    return bool(np.any(values < 0.0) or np.any(values > 1.0))


def adjust_hue_saturation(rgb: np.ndarray, hue_offset: float, saturation_scale: float) -> np.ndarray:
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1.0e-8
    red = nonzero & (maximum == rgb[..., 0])
    green = nonzero & (maximum == rgb[..., 1])
    blue = nonzero & (maximum == rgb[..., 2])
    hue[red] = np.mod((rgb[..., 1][red] - rgb[..., 2][red]) / delta[red], 6.0)
    hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / delta[green] + 2.0
    hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / delta[blue] + 4.0
    hue = np.mod(hue / 6.0 + hue_offset, 1.0)
    saturation = np.where(maximum > 1.0e-8, delta / np.maximum(maximum, 1.0e-8), 0.0)
    saturation = np.clip(saturation * max(0.0, saturation_scale), 0.0, 1.0)

    sector = np.floor(hue * 6.0).astype(np.int32)
    fraction = hue * 6.0 - sector
    p = maximum * (1.0 - saturation)
    q = maximum * (1.0 - fraction * saturation)
    t = maximum * (1.0 - (1.0 - fraction) * saturation)
    result = np.empty_like(rgb)
    components = (
        (maximum, t, p),
        (q, maximum, p),
        (p, maximum, t),
        (p, q, maximum),
        (t, p, maximum),
        (maximum, p, q),
    )
    sector = np.mod(sector, 6)
    for value, channels in enumerate(components):
        mask = sector == value
        for channel, component in enumerate(channels):
            result[..., channel][mask] = component[mask]
    return result


def apply_base_color_controls(
    pixels: np.ndarray,
    material: EffectiveMaterial,
    association: str,
    index: int,
) -> np.ndarray:
    rgb = srgb_to_linear(pixels[..., :3])
    tint = selected_color(
        material,
        ("Base Color Tint", "Diffuse Tint", "Tint"),
        association,
        index,
        (1.0, 1.0, 1.0, 1.0),
    )[:3]
    rgb *= tint
    contrast = material.scalar(("Base Color Contrast",), association, index, 0.0)
    rgb = (rgb - 0.5) * max(0.0, 1.0 + contrast) + 0.5
    saturation = material.scalar(("Saturation",), association, index, 1.0)
    luminance = max(0.0, material.scalar(("Luminance",), association, index, 1.0))
    hue = material.scalar(("Hue",), association, index, 0.5) - 0.5
    # Zorah's parent layer uses a neutral Hue of 0.5.  Apply the same HSV
    # convention to the linear texture sample before returning to sRGB.
    if abs(hue) > 1.0e-6 or abs(saturation - 1.0) > 1.0e-6:
        rgb = adjust_hue_saturation(np.clip(rgb, 0.0, None), hue, saturation)
    rgb *= luminance
    result = np.empty((*rgb.shape[:2], 4), dtype=np.float32)
    result[..., :3] = linear_to_srgb(rgb)
    result[..., 3] = pixels[..., 3] if pixels.shape[-1] > 3 else 1.0
    return result


def image_record(
    object_name: str,
    output: str,
    size: tuple[int, int],
    *,
    srgb: bool,
    normal_map: bool,
    grid: tuple[int, int] = (1, 1),
) -> dict[str, Any]:
    grid_columns, grid_rows = grid
    return {
        "object": object_name,
        "source": "Zorah material bake",
        "source_size": list(size),
        "output": output,
        "output_size": list(size),
        "source_format": "RGBA8",
        "source_compression": "material-bake",
        "srgb": srgb,
        "normal_map": normal_map,
        "source_block_count": grid_columns * grid_rows,
        "source_grid_columns": grid_columns,
        "source_grid_rows": grid_rows,
        "source_payload_size": None,
        "source_consumed_bytes": None,
        "source_payload_prefix_bytes": None,
        "source_payload_tail_bytes": None,
        "recovered_oodle_blocks": [],
        "output_bit_depth": 8,
        "output_file_size": None,
        "material_bake": True,
    }


HIGH_PRECISION_MODES = {"I", "I;16", "I;16L", "I;16B", "I;16N"}


def open_rgba(path: Path) -> Image.Image:
    """Read an exported PNG as 8-bit RGBA.

    Pillow reopens the exporter's 16-bit grayscale PNGs (TSF_G16 sources) as
    "I" or "I;16"; converting those straight to RGBA clips at 255 instead of
    rescaling, which reads a height/mask texture back as solid white.
    """
    with Image.open(path) as source:
        source.load()
        if source.mode not in HIGH_PRECISION_MODES:
            return source.convert("RGBA")
        scaled = np.asarray(source, dtype=np.float32) * (255.0 / 65535.0)
        gray = Image.fromarray(
            np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.uint8), "L"
        )
        return gray.convert("RGBA")


class TextureSet:
    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = root
        self.records = {record["object"]: record for record in manifest["exported"]}

    def record(self, reference: str | None) -> dict[str, Any] | None:
        return self.records.get(reference) if reference else None

    def open(self, reference: str | None) -> Image.Image | None:
        record = self.record(reference)
        if record is None:
            return None
        return open_rgba(self.root / record["output"])

    def grid(self, reference: str | None) -> tuple[int, int]:
        record = self.record(reference)
        if record is None:
            return (1, 1)
        return (
            max(1, int(record.get("source_grid_columns", 1))),
            max(1, int(record.get("source_grid_rows", 1))),
        )


def texture_reference(
    material: EffectiveMaterial,
    names: Iterable[str],
    association: str,
    index: int,
) -> tuple[str | None, str | None]:
    parameter = material.select("textures", names, association, index)
    if parameter is None:
        return None, None
    return str(parameter["value"]), str(parameter["name"])


def uv_controls(
    material: EffectiveMaterial, association: str, index: int, prefix: str = ""
) -> tuple[float, float, float, float, float]:
    name = lambda value: f"{prefix}{value}" if prefix else value
    return (
        material.scalar((name("UV Scale U"), name("UV X")), association, index, 1.0),
        material.scalar((name("UV Scale V"), name("UV Y")), association, index, 1.0),
        material.scalar((name("UV Offset U"), name("UV Offset X")), association, index, 0.0),
        material.scalar((name("UV Offset V"), name("UV Offset Y")), association, index, 0.0),
        material.scalar((name("UV Rotation In Degrees"),), association, index, 0.0),
    )


def neutral_uv_controls(material: EffectiveMaterial, association: str, index: int) -> bool:
    return all(
        math.isclose(actual, expected)
        for actual, expected in zip(
            uv_controls(material, association, index),
            (1.0, 1.0, 0.0, 0.0, 0.0),
        )
    )


def sample_tiled(
    image: Image.Image,
    size: tuple[int, int],
    controls: tuple[float, float, float, float, float],
    *,
    source_grid: tuple[int, int] = (1, 1),
    target_grid: tuple[int, int] = (1, 1),
    is_srgb: bool = False,
    approximations: set[str] | None = None,
) -> np.ndarray:
    source = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    width, height = size
    scale_u, scale_v, offset_u, offset_v, rotation_degrees = controls
    result = np.empty((height, width, 4), dtype=np.float32)
    source_columns, source_rows = source_grid
    target_columns, target_rows = target_grid
    if is_srgb:
        # A GPU sRGB sampler filters in linear light. Decode once, blend, and
        # re-encode so callers still receive sRGB-encoded pixels.
        source[..., :3] = srgb_to_linear(source[..., :3])
    if approximations is not None:
        tile_width = max(1.0, width / target_columns)
        tile_height = max(1.0, height / target_rows)
        minification = max(
            source.shape[1] / source_columns * abs(scale_u) / tile_width,
            source.shape[0] / source_rows * abs(scale_v) / tile_height,
        )
        if minification > 2.0:
            approximations.add("bilinear sampling minifies without a prefilter")
    # Generated atlases cover the target grid's actual UE UV domain. A 2x1
    # output therefore spans U=[0, 2), rather than squeezing two authored
    # tiles into U=[0, 1). One-tile inputs still repeat once per UV tile while
    # multi-tile inputs retain their distinct UDIM pages.
    u = (
        (np.arange(width, dtype=np.float32) + 0.5)
        / width
        * target_columns
        - 0.5
    )
    cosine = math.cos(math.radians(rotation_degrees))
    sine = math.sin(math.radians(rotation_degrees))
    for y0 in range(0, height, 128):
        y1 = min(y0 + 128, height)
        # Both atlases follow the exporter's row order: UDIM row 0 (mesh v in
        # [0, 1]) sits at the bottom, so image V spans mesh v + rows - 1. The
        # UV controls act on mesh v, and the result returns to the source
        # atlas's own row origin - which differs from the target's whenever the
        # two grids have different row counts.
        v = (
            (np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5)
            / height
            * target_rows
            - target_rows
            + 0.5
        )
        rotated_u = cosine * u[None, :] - sine * v
        rotated_v = sine * u[None, :] + cosine * v
        source_u = (
            np.mod(rotated_u * scale_u + 0.5 + offset_u, source_columns)
            / source_columns
            * source.shape[1]
            - 0.5
        )
        source_v = (
            np.mod(
                rotated_v * scale_v + 0.5 + offset_v + source_rows - 1,
                source_rows,
            )
            / source_rows
            * source.shape[0]
            - 0.5
        )
        x_floor = np.floor(source_u).astype(np.int32)
        y_floor = np.floor(source_v).astype(np.int32)
        x_fraction = (source_u - x_floor)[..., None]
        y_fraction = (source_v - y_floor)[..., None]
        x_base = x_floor % source.shape[1]
        y_base = y_floor % source.shape[0]
        x_next = (x_base + 1) % source.shape[1]
        y_next = (y_base + 1) % source.shape[0]
        top = source[y_base, x_base] * (1.0 - x_fraction) + source[y_base, x_next] * x_fraction
        bottom = source[y_next, x_base] * (1.0 - x_fraction) + source[y_next, x_next] * x_fraction
        result[y0:y1] = top * (1.0 - y_fraction) + bottom * y_fraction
    if is_srgb:
        result[..., :3] = linear_to_srgb(result[..., :3])
    return result


def target_layout(
    references: Iterable[str | None], textures: TextureSet, maximum: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    records = [
        textures.records[reference]
        for reference in references
        if reference in textures.records
    ]
    grid = (
        max((max(1, int(record.get("source_grid_columns", 1))) for record in records), default=1),
        max((max(1, int(record.get("source_grid_rows", 1))) for record in records), default=1),
    )
    tile_width = max(
        (
            int(record["output_size"][0])
            / max(1, int(record.get("source_grid_columns", 1)))
            for record in records
        ),
        default=4.0,
    )
    tile_height = max(
        (
            int(record["output_size"][1])
            / max(1, int(record.get("source_grid_rows", 1)))
            for record in records
        ),
        default=4.0,
    )
    width = min(maximum, max(4, math.ceil(tile_width * grid[0])))
    height = min(maximum, max(4, math.ceil(tile_height * grid[1])))
    return (((width + 3) // 4 * 4, (height + 3) // 4 * 4), grid)


def target_size(
    references: Iterable[str | None], textures: TextureSet, maximum: int
) -> tuple[int, int]:
    return target_layout(references, textures, maximum)[0]


def neutral_base_controls(material: EffectiveMaterial, association: str, index: int) -> bool:
    tint = selected_color(
        material,
        ("Base Color Tint", "Diffuse Tint", "Tint"),
        association,
        index,
        (1.0, 1.0, 1.0, 1.0),
    )
    return (
        np.allclose(tint[:3], 1.0, atol=1.0 / 255.0)
        and math.isclose(material.scalar(("Hue",), association, index, 0.5), 0.5)
        and math.isclose(material.scalar(("Saturation",), association, index, 1.0), 1.0)
        and math.isclose(material.scalar(("Luminance",), association, index, 1.0), 1.0)
        and math.isclose(material.scalar(("Base Color Contrast",), association, index, 0.0), 0.0)
    )


def save_pixels(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), "RGBA")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    image.save(temporary, format="PNG", compress_level=1)
    os.replace(temporary, path)


def blend_mask(
    material: EffectiveMaterial,
    blend_index: int,
    textures: TextureSet,
    size: tuple[int, int],
    target_grid: tuple[int, int],
    approximations: set[str],
) -> np.ndarray:
    reference, _ = texture_reference(material, ("Texture Mask",), BLEND, blend_index)
    image = textures.open(reference)
    if image is None:
        approximations.add("missing texture/vertex blend mask uses 0.5")
        return np.full((size[1], size[0]), 0.5, dtype=np.float32)
    controls = (
        material.scalar(("Texture Mask UV X",), BLEND, blend_index, 1.0),
        material.scalar(("Texture Mask UV Y",), BLEND, blend_index, 1.0),
        material.scalar(("Texture Mask UV Offset X",), BLEND, blend_index, 0.0),
        material.scalar(("Texture Mask UV Offset Y",), BLEND, blend_index, 0.0),
        0.0,
    )
    pixels = sample_tiled(
        image,
        size,
        controls,
        source_grid=textures.grid(reference),
        target_grid=target_grid,
        approximations=approximations,
    )
    channel = selected_color(
        material,
        ("Texture Mask Channel Select",),
        BLEND,
        blend_index,
        (1.0, 0.0, 0.0, 0.0),
    )[:3]
    channel_index = int(np.argmax(channel))
    mask = pixels[..., channel_index]
    if material.switch(("Texture Mask Invert",), BLEND, blend_index):
        mask = 1.0 - mask
    contrast = material.scalar(("Texture Mask Contrast",), BLEND, blend_index, 0.0)
    position = material.scalar(("Texture Mask Position",), BLEND, blend_index, 0.0)
    mask = (mask - 0.5) * max(0.0, 1.0 + contrast) + 0.5 + position
    final_contrast = material.scalar(("Final Texture Mask Contrast",), BLEND, blend_index, 0.0)
    final_position = material.scalar(("Final Texture Mask Position",), BLEND, blend_index, 0.0)
    mask = (mask - 0.5) * max(0.0, 1.0 + final_contrast) + 0.5 + final_position
    mask *= max(0.0, material.scalar(("Blend Opacity",), BLEND, blend_index, 1.0))
    if material.switch(("Use Top Down Projection",), BLEND, blend_index):
        approximations.add("top-down world projection baked in UV0")
    if material.switch(("Use UV2",), BLEND, blend_index):
        approximations.add("UV2 blend mask baked in UV0")
    if material.switch(
        ("Toggle Vertex Color", "Toggle Texture Mask / Vertex Color"),
        BLEND,
        blend_index,
    ):
        approximations.add("vertex-color blend contribution unavailable to material-only bake")
    if material.switch(("Toggle Linear / Height Blend",), BLEND, blend_index):
        approximations.add("height blend uses exported texture mask without UE graph height bias")
    return np.clip(mask, 0.0, 1.0)


def layer_pixels(
    material: EffectiveMaterial,
    layer_index: int,
    kind: str,
    textures: TextureSet,
    size: tuple[int, int],
    target_grid: tuple[int, int],
    approximations: set[str],
) -> np.ndarray:
    names = BASE_NAMES if kind == "base" else NORMAL_NAMES if kind == "normal" else SURFACE_NAMES
    reference, parameter_name = texture_reference(material, names, LAYER, layer_index)
    image = textures.open(reference)
    if image is None:
        fill = (255, 255, 255, 255)
        if kind == "normal":
            fill = (128, 128, 255, 255)
        elif kind == "surface":
            fill = (255, 128, 0, 255)
        image = Image.new("RGBA", (4, 4), fill)
        approximations.add(f"layer {layer_index} has no {kind} texture; baked a neutral fill")
    pixels = sample_tiled(
        image,
        size,
        uv_controls(material, LAYER, layer_index),
        source_grid=textures.grid(reference),
        target_grid=target_grid,
        is_srgb=(kind == "base"),
        approximations=approximations,
    )
    if kind == "base":
        return apply_base_color_controls(pixels, material, LAYER, layer_index)
    if kind == "normal":
        vector = pixels[..., :3] * 2.0 - 1.0
        intensity = material.scalar(("Normal Intensity",), LAYER, layer_index, 1.0)
        vector[..., :2] *= intensity
        length = np.linalg.norm(vector, axis=-1, keepdims=True)
        vector /= np.maximum(length, 1.0e-6)
        pixels[..., :3] = vector * 0.5 + 0.5
        pixels[..., 3] = 1.0
        return pixels
    roughness = pixels[..., 1]
    roughness_contrast = material.scalar(("Roughness Contrast",), LAYER, layer_index, 0.0)
    roughness_offset = material.scalar(("Roughness Offset",), LAYER, layer_index, 0.0)
    pixels[..., 0] += material.scalar(("Occlusion Offset",), LAYER, layer_index, 0.0)
    pixels[..., 1] = (roughness - 0.5) * max(0.0, 1.0 + roughness_contrast) + 0.5 + roughness_offset
    if normalized(parameter_name or "") == "ors":
        pixels[..., 2] = 0.0
    pixels[..., 3] = 1.0
    return np.clip(pixels, 0.0, 1.0)


def composite_layers(
    material: EffectiveMaterial,
    kind: str,
    textures: TextureSet,
    size: tuple[int, int],
    target_grid: tuple[int, int],
    masks: list[np.ndarray],
    approximations: set[str],
) -> np.ndarray:
    result = layer_pixels(material, 0, kind, textures, size, target_grid, approximations)
    for layer_index, mask in enumerate(masks, start=1):
        # Zorah uses this material-attributes layer as an explicit no-op. It
        # forwards the material below it and intentionally has no Base Color,
        # Normal, or ORM texture of its own. Treating those absent textures as
        # neutral fallback images would instead blend white/flat data over the
        # authored photogrammetry.
        if material.layers[layer_index] == PASSTHROUGH_LAYER:
            continue
        top = layer_pixels(
            material, layer_index, kind, textures, size, target_grid, approximations
        )
        weight = mask[..., None]
        if kind == "base":
            bottom_linear = srgb_to_linear(result[..., :3])
            top_linear = srgb_to_linear(top[..., :3])
            result[..., :3] = linear_to_srgb(bottom_linear * (1.0 - weight) + top_linear * weight)
            result[..., 3] = 1.0
        elif kind == "normal":
            bottom_vector = result[..., :3] * 2.0 - 1.0
            top_vector = top[..., :3] * 2.0 - 1.0
            vector = bottom_vector * (1.0 - weight) + top_vector * weight
            vector /= np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1.0e-6)
            result[..., :3] = vector * 0.5 + 0.5
            result[..., 3] = 1.0
        else:
            result = result * (1.0 - weight) + top * weight
    return np.clip(result, 0.0, 1.0)


def runtime_parameter(name: str, value: Any) -> dict[str, Any]:
    return {"name": name, "association": GLOBAL, "index": -1, "value": value}


def emissive_properties(
    material: EffectiveMaterial,
) -> tuple[bool, float, np.ndarray, tuple[str, str] | None]:
    switches = [
        parameter
        for parameter in material.switches.values()
        if normalized(str(parameter.get("name", ""))) == "enableemissive"
    ]
    enabled_switches = [parameter for parameter in switches if parameter.get("value") is True]
    if enabled_switches:
        selected_switch = sorted(
            enabled_switches,
            key=lambda parameter: (
                str(parameter.get("association") or GLOBAL) != GLOBAL,
                int(parameter.get("index", -1)),
            ),
        )[0]
        association = str(selected_switch.get("association") or GLOBAL)
        index = int(selected_switch.get("index", -1))
    elif switches:
        return False, 0.0, np.ones(4, dtype=np.float32), None
    elif material.select("scalars", ("Global Emission Intensity",), GLOBAL, -1):
        # Zorah's custom spherical-lamp graph is unconditionally emissive and
        # uses this distinct parameter family rather than an enable switch.
        association, index = GLOBAL, -1
    elif material.object.startswith("/Game/VFX/") and material.select(
        "scalars", ("Emissive Intensity", "EmissiveIntensity"), GLOBAL, -1
    ):
        # The sample's custom unlit VFX graphs wire emission directly. Generic
        # LS parent materials instead require the Enable Emissive static switch.
        association, index = GLOBAL, -1
    else:
        return False, 0.0, np.ones(4, dtype=np.float32), None

    intensity = material.scalar(
        EMISSIVE_INTENSITY_NAMES, association, index, 1.0
    )
    if intensity <= 0.0:
        return False, 0.0, np.ones(4, dtype=np.float32), None
    color = selected_color(
        material,
        EMISSIVE_COLOR_NAMES,
        association,
        index,
        (1.0, 1.0, 1.0, 1.0),
    )
    reference, parameter_name = texture_reference(
        material, EMISSIVE_TEXTURE_NAMES, association, index
    )
    texture = (
        (parameter_name or "Emissive", reference)
        if reference is not None
        else None
    )
    return True, intensity, color, texture


def apply_runtime_emissive(
    runtime: dict[str, Any],
    material: EffectiveMaterial,
    approximations: set[str] | None = None,
) -> dict[str, Any]:
    runtime = dict(runtime)
    runtime["scalars"] = [
        parameter
        for parameter in runtime.get("scalars", [])
        if normalized(str(parameter.get("name", "")))
        not in {normalized(name) for name in EMISSIVE_INTENSITY_NAMES}
    ]
    runtime["vectors"] = [
        parameter
        for parameter in runtime.get("vectors", [])
        if normalized(str(parameter.get("name", "")))
        not in {normalized(name) for name in EMISSIVE_COLOR_NAMES}
    ]
    runtime["textures"] = [
        parameter
        for parameter in runtime.get("textures", [])
        if normalized(str(parameter.get("name", "")))
        not in {normalized(name) for name in EMISSIVE_TEXTURE_NAMES}
    ]
    enabled, intensity, color, texture = emissive_properties(material)
    runtime["emissive"] = enabled
    if not enabled:
        return runtime
    peak = float(np.max(color[:3]))
    if peak > 1.0:
        # The runtime multiplies the 8-bit color by this scalar, so an HDR
        # authored color keeps its magnitude once the peak moves into intensity.
        color = np.asarray(color, dtype=np.float32).copy()
        color[:3] /= peak
        intensity *= peak
    if approximations is not None and out_of_range_components(color):
        approximations.add("emissive color clamped to the 0..1 hex range")
    runtime["scalars"].append(runtime_parameter("Emissive Intensity", intensity))
    runtime["vectors"].append(
        runtime_parameter("Emissive Color", f"{hex_color(color)} (FLinearColor)")
    )
    if texture is not None:
        _, reference = texture
        runtime["textures"].append(runtime_parameter("Emissive", reference))
    return runtime


def runtime_material_record(
    material: EffectiveMaterial,
    textures: dict[str, tuple[str, str]],
    approximations: set[str] | None = None,
) -> dict[str, Any]:
    scalar_parameters: list[dict[str, Any]] = []
    vector_parameters: list[dict[str, Any]] = []
    texture_parameters = [runtime_parameter(name, reference) for name, reference in textures.values()]
    metallic = material.scalar(("Metallic", "Metalness"), GLOBAL, -1, math.nan)
    roughness = material.scalar(("Roughness",), GLOBAL, -1, math.nan)
    if math.isfinite(metallic):
        scalar_parameters.append(runtime_parameter("Metallic", metallic))
    if math.isfinite(roughness):
        scalar_parameters.append(runtime_parameter("Roughness", roughness))
    if "base" not in textures:
        tint = selected_color(
            material,
            ("Base Color Tint", "Diffuse Tint", "Tint", "Base Color"),
            GLOBAL,
            -1,
            (1.0, 1.0, 1.0, 1.0),
        )
        if approximations is not None and out_of_range_components(tint):
            approximations.add("base color tint clamped to the 0..1 hex range")
        vector_parameters.append(
            runtime_parameter("Tint", f"{hex_color(tint)} (FLinearColor)")
        )
        luminance = material.scalar(("Luminance",), GLOBAL, -1, 1.0)
        if not math.isclose(luminance, 1.0):
            scalar_parameters.append(runtime_parameter("Luminance", luminance))
    return apply_runtime_emissive({
        "package": material.package,
        "object": material.object,
        "type": material.type,
        "parent": None,
        "scalars": scalar_parameters,
        "vectors": vector_parameters,
        "textures": texture_parameters,
        "static_switches": [],
        "layers": [],
        "blends": [],
        "base_overrides": material.base_overrides,
    }, material, approximations)


def bake_material(
    material: EffectiveMaterial,
    texture_set: TextureSet,
    output_root: Path,
    maximum_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    approximations: set[str] = set()
    selected: dict[str, tuple[str, str]] = {}
    generated: list[dict[str, Any]] = []
    layer_count = len(material.layers)
    digest = hashlib.sha256(material.object.encode("utf-8")).hexdigest()[:16]

    if layer_count:
        references = [
            texture_reference(material, names, LAYER, index)[0]
            for index in range(layer_count)
            for names in (BASE_NAMES, NORMAL_NAMES, SURFACE_NAMES)
        ] + [
            texture_reference(material, ("Texture Mask",), BLEND, index)[0]
            for index in range(max(0, layer_count - 1))
        ]
        size, grid = target_layout(references, texture_set, maximum_size)
        masks = [
            np.zeros((size[1], size[0]), dtype=np.float32)
            if material.layers[index + 1] == PASSTHROUGH_LAYER
            else blend_mask(material, index, texture_set, size, grid, approximations)
            for index in range(layer_count - 1)
        ]
        for kind, parameter_name, srgb, normal_map in (
            ("base", "Base Color", True, False),
            ("normal", "Normal", False, True),
            ("surface", "ORM", False, False),
        ):
            relative = f"MaterialBakes/{digest}_{kind}.png"
            path = output_root / relative
            save_pixels(
                path,
                composite_layers(
                    material, kind, texture_set, size, grid, masks, approximations
                ),
            )
            object_name = f"/ZorahGenerated/MaterialBakes/{digest}_{kind}"
            record = image_record(
                object_name,
                relative,
                size,
                srgb=srgb,
                normal_map=normal_map,
                grid=grid,
            )
            record["output_file_size"] = path.stat().st_size
            generated.append(record)
            selected[kind] = (parameter_name, object_name)
    else:
        selected_scopes: dict[str, tuple[str, int]] = {}
        source_names: dict[str, str] = {}
        for kind, names in (
            ("base", BASE_NAMES),
            ("normal", NORMAL_NAMES),
            ("surface", SURFACE_NAMES),
            ("emissive", EMISSIVE_NAMES),
        ):
            # Some Zorah instances contain a flattened material layer's
            # parameters but no layer-stack metadata. UE still evaluates the
            # LayerParameter[0] values in that case.
            for association, index in ((GLOBAL, -1), (LAYER, 0)):
                reference, source_name = texture_reference(
                    material, names, association, index
                )
                if reference is not None and reference in texture_set.records:
                    selected[kind] = (RUNTIME_TEXTURE_NAMES[kind], reference)
                    selected_scopes[kind] = (association, index)
                    source_names[kind] = source_name or ""
                    break

        base_reference = selected.get("base", (None, None))[1]
        base_scope = selected_scopes.get("base", (GLOBAL, -1))
        if base_reference and (
            not neutral_base_controls(material, *base_scope)
            or not neutral_uv_controls(material, *base_scope)
        ):
            size, grid = target_layout((base_reference,), texture_set, maximum_size)
            image = texture_set.open(base_reference)
            assert image is not None
            pixels = apply_base_color_controls(
                sample_tiled(
                    image,
                    size,
                    uv_controls(material, *base_scope),
                    source_grid=texture_set.grid(base_reference),
                    target_grid=grid,
                    is_srgb=True,
                    approximations=approximations,
                ),
                material,
                *base_scope,
            )
            relative = f"MaterialBakes/{digest}_base.png"
            path = output_root / relative
            save_pixels(path, pixels)
            object_name = f"/ZorahGenerated/MaterialBakes/{digest}_base"
            record = image_record(
                object_name,
                relative,
                size,
                srgb=True,
                normal_map=False,
                grid=grid,
            )
            record["output_file_size"] = path.stat().st_size
            generated.append(record)
            selected["base"] = ("Base Color", object_name)

        normal_reference = selected.get("normal", (None, None))[1]
        normal_scope = selected_scopes.get("normal", (GLOBAL, -1))
        normal_intensity = material.scalar(("Normal Intensity",), *normal_scope, 1.0)
        if normal_reference and (
            not math.isclose(normal_intensity, 1.0)
            or not neutral_uv_controls(material, *normal_scope)
        ):
            size, grid = target_layout((normal_reference,), texture_set, maximum_size)
            image = texture_set.open(normal_reference)
            assert image is not None
            pixels = sample_tiled(
                image,
                size,
                uv_controls(material, *normal_scope),
                source_grid=texture_set.grid(normal_reference),
                target_grid=grid,
                approximations=approximations,
            )
            vector = pixels[..., :3] * 2.0 - 1.0
            vector[..., :2] *= normal_intensity
            vector /= np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1.0e-6)
            pixels[..., :3] = vector * 0.5 + 0.5
            pixels[..., 3] = 1.0
            relative = f"MaterialBakes/{digest}_normal.png"
            path = output_root / relative
            save_pixels(path, pixels)
            object_name = f"/ZorahGenerated/MaterialBakes/{digest}_normal"
            record = image_record(
                object_name,
                relative,
                size,
                srgb=False,
                normal_map=True,
                grid=grid,
            )
            record["output_file_size"] = path.stat().st_size
            generated.append(record)
            selected["normal"] = ("Normal", object_name)

        surface_reference = selected.get("surface", (None, None))[1]
        surface_scope = selected_scopes.get("surface", (GLOBAL, -1))
        roughness_offset = material.scalar(("Roughness Offset",), *surface_scope, 0.0)
        roughness_contrast = material.scalar(("Roughness Contrast",), *surface_scope, 0.0)
        occlusion_offset = material.scalar(("Occlusion Offset",), *surface_scope, 0.0)
        source_surface_name = source_names.get("surface", "")
        if surface_reference and (
            not math.isclose(roughness_offset, 0.0)
            or not math.isclose(roughness_contrast, 0.0)
            or not math.isclose(occlusion_offset, 0.0)
            or normalized(source_surface_name) == "ors"
            or not neutral_uv_controls(material, *surface_scope)
        ):
            size, grid = target_layout((surface_reference,), texture_set, maximum_size)
            image = texture_set.open(surface_reference)
            assert image is not None
            pixels = sample_tiled(
                image,
                size,
                uv_controls(material, *surface_scope),
                source_grid=texture_set.grid(surface_reference),
                target_grid=grid,
                approximations=approximations,
            )
            pixels[..., 0] += occlusion_offset
            pixels[..., 1] = (
                (pixels[..., 1] - 0.5) * max(0.0, 1.0 + roughness_contrast)
                + 0.5
                + roughness_offset
            )
            if normalized(source_surface_name) == "ors":
                pixels[..., 2] = 0.0
            pixels[..., 3] = 1.0
            relative = f"MaterialBakes/{digest}_surface.png"
            path = output_root / relative
            save_pixels(path, np.clip(pixels, 0.0, 1.0))
            object_name = f"/ZorahGenerated/MaterialBakes/{digest}_surface"
            record = image_record(
                object_name,
                relative,
                size,
                srgb=False,
                normal_map=False,
                grid=grid,
            )
            record["output_file_size"] = path.stat().st_size
            generated.append(record)
            selected["surface"] = ("ORM", object_name)

    return (
        runtime_material_record(material, selected, approximations),
        generated,
        approximations,
    )


def signature_for(
    material: EffectiveMaterial, texture_set: TextureSet, maximum_size: int
) -> str:
    references = sorted(
        str(parameter.get("value"))
        for parameter in material.textures.values()
        if parameter.get("value") in texture_set.records
    )
    source_state = []
    for reference in references:
        record = texture_set.records[reference]
        path = texture_set.root / record["output"]
        stat = path.stat()
        source_state.append((reference, record.get("output_size"), stat.st_size, stat.st_mtime_ns))
    payload = {
        "version": (
            LAYER_MATERIAL_BAKE_PIPELINE_VERSION
            if material.layers
            else MATERIAL_BAKE_PIPELINE_VERSION
        ),
        "maximum_size": maximum_size,
        "material": {
            "object": material.object,
            "layers": material.layers,
            "blends": material.blends,
            "scalars": sorted(material.scalars.values(), key=parameter_key),
            "vectors": sorted(material.vectors.values(), key=parameter_key),
            "textures": sorted(material.textures.values(), key=parameter_key),
            "switches": sorted(material.switches.values(), key=parameter_key),
            "base_overrides": material.base_overrides,
        },
        "sources": source_state,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_materials", type=Path)
    parser.add_argument("source_textures", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="directory containing source texture outputs (defaults to output_root)",
    )
    parser.add_argument("--max-size", type=int, default=4096)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="log every material record")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="bake only the named material object (repeatable; diagnostic use)",
    )
    args = parser.parse_args()
    if args.max_size <= 0 or args.max_size % 4:
        parser.error("--max-size must be a positive multiple of four")
    if args.jobs < 1:
        parser.error("--jobs must be at least one")

    material_manifest = load_json(args.source_materials)
    texture_manifest = load_json(args.source_textures)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    texture_set = TextureSet(
        args.source_root.resolve() if args.source_root else output_root,
        texture_manifest,
    )
    resolver = Resolver(material_manifest)
    previous_path = output_root / "material_bakes.json"
    previous = load_json(previous_path) if args.resume and previous_path.is_file() else {}
    previous_entries = previous.get("materials", {})

    runtime_materials: list[dict[str, Any]] = []
    generated_textures: list[dict[str, Any]] = []
    report_entries: dict[str, Any] = {}
    reused = 0
    baked = 0
    requested = (
        list(args.only)
        if args.only
        else list(material_manifest.get("requested", []))
    )
    work = []
    for object_name in requested:
        material = resolver.resolve(object_name)
        signature = signature_for(material, texture_set, args.max_size)
        prior = previous_entries.get(object_name)
        can_reuse = bool(
            prior
            and prior.get("signature") == signature
            and all((output_root / record["output"]).is_file() for record in prior.get("textures", []))
        )
        work.append((object_name, material, signature, prior, can_reuse))

    def process_material(item):
        object_name, material, signature, prior, can_reuse = item
        if can_reuse:
            runtime = apply_runtime_emissive(prior["runtime_material"], material)
            textures = prior.get("textures", [])
            approximations = set(prior.get("approximations", []))
        else:
            runtime, textures, approximations = bake_material(
                material, texture_set, output_root, args.max_size
            )
        return (
            object_name,
            material,
            signature,
            can_reuse,
            runtime,
            textures,
            approximations,
        )

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = executor.map(process_material, work)
        for index, result in enumerate(results, start=1):
            (
                object_name,
                material,
                signature,
                can_reuse,
                runtime,
                textures,
                approximations,
            ) = result
            reused += int(can_reuse)
            baked += int(not can_reuse)
            runtime_materials.append(runtime)
            generated_textures.extend(textures)
            report_entries[object_name] = {
                "signature": signature,
                "layer_count": len(material.layers),
                "blend_count": len(material.blends),
                "runtime_material": runtime,
                "textures": textures,
                "approximations": sorted(approximations),
            }
            if args.verbose:
                print(
                    f"ZORAH_MATERIAL_BAKE {index}/{len(requested)} object={object_name} "
                    f"layers={len(material.layers)} generated={len(textures)} reused={can_reuse}",
                    flush=True,
                )

    referenced = {
        parameter["value"]
        for material in runtime_materials
        for parameter in material.get("textures", [])
        if parameter.get("value")
    }
    all_texture_records = dict(texture_set.records)
    all_texture_records.update({record["object"]: record for record in generated_textures})
    missing = referenced - all_texture_records.keys()
    if missing:
        raise ValueError(f"runtime materials reference {len(missing)} missing textures: {sorted(missing)[:5]}")
    runtime_textures = [all_texture_records[reference] for reference in sorted(referenced)]

    runtime_material_manifest = {
        "format": "zorah-runtime-material-manifest-v1",
        "engine_version": material_manifest.get("engine_version", "5.4"),
        "requested": requested,
        "materials": runtime_materials,
        "texture_references": sorted(referenced),
        "failures": [],
    }
    runtime_texture_manifest = {
        "format": "zorah-runtime-texture-export-v1",
        "source_manifest": str(args.source_textures),
        "max_size": args.max_size,
        "exported": runtime_textures,
        "failures": [],
    }
    report = {
        "format": "zorah-material-bake-v1",
        "max_size": args.max_size,
        "source_materials": str(args.source_materials),
        "source_textures": str(args.source_textures),
        "material_count": len(runtime_materials),
        "generated_texture_count": len(generated_textures),
        "runtime_texture_count": len(runtime_textures),
        "baked_this_run": baked,
        "reused_this_run": reused,
        "materials": report_entries,
    }
    write_json_atomic(output_root / "materials.runtime.json", runtime_material_manifest)
    write_json_atomic(output_root / "textures.runtime.json", runtime_texture_manifest)
    write_json_atomic(previous_path, report)
    print(
        f"ZORAH_MATERIAL_BAKE_DONE materials={len(runtime_materials)} "
        f"generated_textures={len(generated_textures)} runtime_textures={len(runtime_textures)} "
        f"baked={baked} reused={reused}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
