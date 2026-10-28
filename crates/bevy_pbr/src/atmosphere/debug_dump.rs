//! Debug: GPU texture readback for cloud validation.
//!
//! `CLOUD_DUMP_DIR=<dir>` + `CLOUD_DUMP_FRAMES=4,300` copies the listed textures to buffers on
//! those frames and writes raw RGBA16F/R32F files `<dir>/<name>_f<frame>_<w>x<h>x<d>.<fmt>` (rows
//! padded to 256 bytes; the padded width is in the file name). Frames are [`FrameCount`]s, so
//! every view dumps on the same frame (callers tag names with the view); copies are mapped in a
//! later frame, after their submission (a per-invocation counter mapped an earlier view's copies
//! during a later view of the same frame, before they were submitted: audit 31).
//!
//! Each dump directory gets a `manifest.txt` (Q23 provenance): the commit and whether the tree
//! had uncommitted changes, every `ATMOSPHERE_*`, `CLOUD_*` and `BEVY_*` variable, the dump
//! frames, and the frames at which reference accumulators (re)started, from which a reference
//! dump's samples per pixel follow: (dump frame - start - 120) / 4 (each half-res pixel is
//! traced every 4th frame, after 120 settling frames).
//!
//! `CLOUD_DUMP_ONLY=clouds_recon,clouds_trace` writes only the textures whose names start with
//! one of the listed prefixes (long multi-frame dumps of the full set fill the disk).
//!
//! [`FrameCount`]: bevy_diagnostic::FrameCount

use bevy_math::UVec3;
use bevy_render::{
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, CommandEncoder, MapMode, Origin3d,
        TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
    },
    renderer::RenderDevice,
};
use std::sync::Mutex;

static PENDING: Mutex<Vec<(Buffer, String, u32)>> = Mutex::new(Vec::new());

/// Appends a line to the dump directory's manifest (written on first use), if dumping.
pub fn note(line: &str) {
    use std::io::Write;
    let Ok(dir) = std::env::var("CLOUD_DUMP_DIR") else {
        return;
    };
    static STARTED: std::sync::Once = std::sync::Once::new();
    let path = std::path::Path::new(&dir).join("manifest.txt");
    STARTED.call_once(|| {
        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        };
        let mut header = format!(
            "commit: {}\nuncommitted changes: {}\n",
            git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
            git(&["status", "--porcelain", "--untracked-files=no"]).map_or("unknown".into(), |s| (!s.is_empty()).to_string()),
        );
        let mut vars: Vec<_> = std::env::vars()
            .filter(|(k, _)| ["ATMOSPHERE_", "CLOUD_", "BEVY_"].iter().any(|p| k.starts_with(p)))
            .collect();
        vars.sort();
        for (k, v) in vars {
            header += &format!("env {k}={v}\n");
        }
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(&path, header);
    });
    if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(&path) {
        let _ = writeln!(file, "{line}");
    }
}

/// Returns `frame` if a dump is requested on it; maps the copies of earlier frames.
pub fn begin_frame(frame: u32) -> Option<u32> {
    let dir = std::env::var("CLOUD_DUMP_DIR").ok()?;
    let mut pending = PENDING.lock().unwrap();
    pending.retain(|(buffer, file, f)| {
        if *f >= frame {
            return true;
        }
        let b = buffer.clone();
        let file = format!("{dir}/{file}");
        buffer.slice(..).map_async(MapMode::Read, move |r| {
            if r.is_ok() {
                let data = b.slice(..).get_mapped_range().map(|v| v.to_vec()).unwrap_or_default();
                let _ = std::fs::write(&file, data);
                println!("cloud dump: {file}");
            }
        });
        false
    });
    let frames = std::env::var("CLOUD_DUMP_FRAMES").unwrap_or_else(|_| "4,300".into());
    let dumping = frames
        .split(',')
        .filter_map(|v| v.trim().parse::<u32>().ok())
        .any(|f| f == frame);
    if dumping {
        note(&format!("dump frame {frame}"));
    }
    dumping.then_some(frame)
}

/// Copies mip `mip` of `texture` (size already at that mip) into a readback buffer.
pub fn dump(
    device: &RenderDevice,
    encoder: &mut CommandEncoder,
    frame: u32,
    name: &str,
    texture: &Texture,
    mip: u32,
    size: UVec3,
    bytes_per_texel: u32,
    ext: &str,
) {
    if let Ok(only) = std::env::var("CLOUD_DUMP_ONLY")
        && !only.split(',').any(|p| !p.is_empty() && name.starts_with(p.trim()))
    {
        return;
    }
    let row = (size.x * bytes_per_texel).div_ceil(256) * 256;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("cloud_debug_dump"),
        size: (row * size.y * size.z) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: mip,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(size.y),
            },
        },
        bevy_render::render_resource::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: size.z,
        },
    );
    let file = format!(
        "{name}_f{frame:03}_{}x{}x{}.{ext}",
        row / bytes_per_texel,
        size.y,
        size.z
    );
    PENDING.lock().unwrap().push((buffer, file, frame));
}
