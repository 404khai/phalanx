//! Cross-implementation `RoPE` validator binary.
//!
//! Reads a work directory produced by Odyssey `scripts/validate_rope.py`:
//!
//! - `manifest.json` — shapes + `RoPE` hyperparameters
//! - `q_in.bin` / `k_in.bin` — row-major f32 little-endian
//!
//! Writes `q_out.bin` / `k_out.bin` and `phalanx_result.json`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use phalanx::{Rope, RopeConfig, RopeScaling, Shape, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Manifest {
    shape: Vec<usize>,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    scale: f32,
    max_position: usize,
    position_offset: usize,
}

#[derive(Debug, Serialize)]
struct PhalanxResult {
    status: String,
    shape: Vec<usize>,
    q_out: String,
    k_out: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("validate_rope error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: validate_rope <work_dir>")?;

    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(dir.join("manifest.json")).context("read manifest.json")?,
    )
    .context("parse manifest.json")?;

    let rope_cfg = RopeConfig {
        dimension_count: u32::try_from(manifest.rotary_dim)?,
        freq_base: manifest.theta,
        scaling: if (manifest.scale - 1.0).abs() < f32::EPSILON {
            None
        } else {
            Some(RopeScaling {
                scaling_type: "linear".into(),
                factor: Some(manifest.scale),
            })
        },
    };

    let rope = Rope::from_rope_config(&rope_cfg, manifest.head_dim, manifest.max_position)
        .context("build Rope")?;

    let q_in = read_tensor(&dir.join("q_in.bin"), &manifest.shape)?;
    let k_in = read_tensor(&dir.join("k_in.bin"), &manifest.shape)?;
    let q_out = rope
        .forward(&q_in, manifest.position_offset)
        .context("rope forward Q")?;
    let k_out = rope
        .forward(&k_in, manifest.position_offset)
        .context("rope forward K")?;

    write_f32_bin(&dir.join("q_out.bin"), q_out.as_slice())?;
    write_f32_bin(&dir.join("k_out.bin"), k_out.as_slice())?;

    let result = PhalanxResult {
        status: "ok".into(),
        shape: manifest.shape,
        q_out: "q_out.bin".into(),
        k_out: "k_out.bin".into(),
    };
    fs::write(
        dir.join("phalanx_result.json"),
        serde_json::to_string_pretty(&result)? + "\n",
    )?;
    println!("phalanx validate_rope: wrote q_out.bin / k_out.bin");
    Ok(())
}

fn read_tensor(path: &Path, shape: &[usize]) -> Result<Tensor> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} length {} not multiple of 4", path.display(), bytes.len());
    }
    let mut data = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        bail!(
            "{} has {} f32 values, expected {} for shape {:?}",
            path.display(),
            data.len(),
            expected,
            shape
        );
    }
    if shape.len() != 2 && shape.len() != 3 {
        bail!(
            "validate_rope expects rank 2 or 3 tensors (got {shape:?}); squeeze batch in Python"
        );
    }
    let tensor_shape = Shape::new(shape.to_vec())?;
    Ok(Tensor::from_vec(data, tensor_shape)?)
}

fn write_f32_bin(path: &Path, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
