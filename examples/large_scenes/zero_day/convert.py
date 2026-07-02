"""Re-export an ORCA Zero-Day FBX as a single self-contained glTF binary (.glb) for
the `zero_day` Bevy example.

Zero-Day (NVIDIA ORCA) ships as `MEASURE_ONE/MEASURE_ONE.fbx` plus a `tex/` folder of
`.dds` textures. Bevy can't load FBX, and -- more importantly -- Blender's FBX
importer mis-reads this Octane-exported asset's material conventions (it routes the
ORM map into `KHR_materials_specular`, turns the BaseColor opacity into alpha blend,
and drops most emissive maps). So instead of trusting the imported material graph,
this rebuilds every material from the naming convention documented in the download's
README:

    <name>_BaseColor.dds  RGB = base color            (sRGB)
    <name>_Specular.dds   R = occlusion, G = roughness, B = metallic (Non-Color, ORM)
    <name>_Normal.dds     DirectX normal map          (Non-Color)
    <name>_Emissive.dds   RGB = emissive color        (sRGB)

Roughness (G) + metallic (B) from the shared `_Specular` image pack into a single
glTF `metallicRoughnessTexture`. Normals are left DirectX-convention and green-flipped
in the Bevy example (`flip_normal_map_y`), matching how the `bistro` example handles
Sponza. Occlusion (the R channel) is left out -- it needs the fragile glTF-settings
node group and contributes little.

Usage:
    blender --background --python convert.py -- <input.fbx> <output.glb>
"""

import glob
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if len(argv) != 2:
    raise SystemExit(
        "usage: blender --background --python convert.py -- <input.fbx> <output.glb>"
    )
src, dst = argv
texdir = os.path.join(os.path.dirname(src), "tex")

# Start from an empty scene and import the FBX (for geometry, material assignment,
# and names). We rebuild the materials below, so we only need Blender's import for
# the mesh <-> material mapping.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src, use_image_search=True)

# Index the texture set by base name: {base_lower: {channel_lower: path}}.
tex = {}
for path in glob.glob(os.path.join(texdir, "*.dds")):
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_" not in stem:
        continue
    base, chan = stem.rsplit("_", 1)
    tex.setdefault(base.lower(), {})[chan.lower()] = path


def base_for_material(mat):
    """Find the texture base name for a material, preferring the BaseColor image
    Blender already linked, then falling back to the material name."""
    if mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image and node.image.filepath:
                stem = os.path.splitext(os.path.basename(node.image.filepath))[0]
                if "_" in stem:
                    base = stem.rsplit("_", 1)[0].lower()
                    if base in tex:
                        return base
    nm = mat.name.lower()
    # The full name, then the name with any trailing "_suffix" (e.g. Blender's "_c4d")
    # stripped; dict.fromkeys keeps order and drops the duplicate when they are equal.
    for cand in dict.fromkeys((nm, nm.rsplit("_", 1)[0])):
        if cand in tex:
            return cand
    return None


def load_image(path, non_color):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def rebuild(mat, base):
    channels = tex[base]
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if "basecolor" in channels:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(channels["basecolor"], non_color=False)
        # Link Color only (not Alpha): the BaseColor alpha is opacity, and blending
        # every opaque surface is what wrecked the depth sort. Keep materials opaque.
        nt.links.new(n.outputs["Color"], bsdf.inputs["Base Color"])

    if "specular" in channels:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(channels["specular"], non_color=True)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(n.outputs["Color"], sep.inputs["Color"])
        # ORM: G -> roughness, B -> metallic (shared image packs into one glTF map).
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

    if "normal" in channels:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(channels["normal"], non_color=True)
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(n.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    if "emissive" in channels:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(channels["emissive"], non_color=False)
        nt.links.new(n.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.0

    # Keep the material opaque (see BaseColor note above).
    try:
        mat.blend_method = "OPAQUE"
    except (AttributeError, TypeError):
        pass


rebuilt = 0
skipped = []
for mat in bpy.data.materials:
    base = base_for_material(mat)
    if base is None:
        skipped.append(mat.name)
        continue
    rebuild(mat, base)
    rebuilt += 1
print("ZERO_DAY_MATERIALS rebuilt=%d skipped=%d" % (rebuilt, len(skipped)))
if skipped:
    print("ZERO_DAY_SKIPPED", skipped[:20])

# Report the world-space mesh bounds so the example's scale/camera can be tuned.
lo = mathutils.Vector((1e30, 1e30, 1e30))
hi = -lo
for obj in bpy.context.scene.objects:
    if obj.type != "MESH":
        continue
    for corner in obj.bound_box:
        v = obj.matrix_world @ mathutils.Vector(corner)
        lo = mathutils.Vector((min(lo.x, v.x), min(lo.y, v.y), min(lo.z, v.z)))
        hi = mathutils.Vector((max(hi.x, v.x), max(hi.y, v.y), max(hi.z, v.z)))
print(
    "ZERO_DAY_BOUNDS min=(%.2f, %.2f, %.2f) max=(%.2f, %.2f, %.2f)"
    % (lo.x, lo.y, lo.z, hi.x, hi.y, hi.z)
)

# The actions run past the scene's playback range (the camera goes to frame 412 but
# the scene ends at 250), so extend the range to cover every action. Otherwise the
# baked animation -- and the flythrough -- gets cut off early.
max_frame = 1
for obj in bpy.context.scene.objects:
    ad = obj.animation_data
    if ad and ad.action:
        max_frame = max(max_frame, int(round(ad.action.frame_range[1])))
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = max_frame
print("ZERO_DAY_FRAME_RANGE 1..%d" % max_frame)

# Export a single self-contained .glb. glTF is Y-up; the scene has no real lights,
# but it carries the film's animated camera (`DynamicCamera2`) and ~550 animated
# objects. `export_animation_mode="SCENE"` bakes them into ONE clip over the scene
# frame range, so every object stays on the film's shared 13.7 s timeline -- exporting
# per-action instead gives each object its own duration, and looping them makes the
# short ones race.
bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format="GLB",
    export_yup=True,
    export_cameras=True,
    export_lights=False,
    export_animations=True,
    export_animation_mode="SCENE",
    export_apply=True,
)
print("ZERO_DAY_EXPORT_DONE", dst)
