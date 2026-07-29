//! Cross-implementation attention validator binary.
//!
//! Reads a work directory from Odyssey `scripts/validate_attention.py`:
//!
//! - `manifest.json` — shapes / head layout / optional `RoPE` flag
//! - `x_in.bin`, `w_q.bin`, `w_k.bin`, `w_v.bin`, `w_o.bin` — row-major f32 LE
//!
//! Writes `y_out.bin` and `phalanx_result.json`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use phalanx::model::RopeConfig;
use phalanx::{Attention, Rope, Shape, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Manifest {
    shape: Vec<usize>,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    #[serde(default)]
    apply_rope: bool,
    #[serde(default = "default_theta")]
    rope_theta: f32,
    #[serde(default)]
    position_offset: usize,
}

fn default_theta() -> f32 {
    10_000.0
}

#[derive(Debug, Serialize)]
struct PhalanxResult {
    status: String,
    shape: Vec<usize>,
    y_out: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("validate_attention error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: validate_attention <work_dir>")?;

    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(dir.join("manifest.json")).context("read manifest.json")?,
    )
    .context("parse manifest.json")?;

    let q_out = manifest.num_heads * manifest.head_dim;
    let kv_out = manifest.num_kv_heads * manifest.head_dim;
    let hidden = manifest.hidden_size;

    let w_q = read_tensor(&dir.join("w_q.bin"), &[q_out, hidden])?;
    let w_k = read_tensor(&dir.join("w_k.bin"), &[kv_out, hidden])?;
    let w_v = read_tensor(&dir.join("w_v.bin"), &[kv_out, hidden])?;
    let w_o = read_tensor(&dir.join("w_o.bin"), &[hidden, q_out])?;

    let attn = Attention::from_tensors(
        w_q,
        w_k,
        w_v,
        w_o,
        manifest.num_heads,
        manifest.num_kv_heads,
        manifest.head_dim,
    )
    .context("build Attention")?;

    let x_in = read_tensor(&dir.join("x_in.bin"), &manifest.shape)?;

    let rope = if manifest.apply_rope {
        let max_pos = if manifest.shape.len() >= 2 {
            manifest.position_offset + manifest.shape[manifest.shape.len() - 2]
        } else {
            64
        };
        let max_pos = max_pos.max(64);
        Some(
            Rope::from_rope_config(
                &RopeConfig {
                    dimension_count: u32::try_from(manifest.head_dim)
                        .context("head_dim does not fit u32")?,
                    freq_base: manifest.rope_theta,
                    scaling: None,
                },
                manifest.head_dim,
                max_pos,
            )
            .context("build Rope")?,
        )
    } else {
        None
    };

    let y_out = attn
        .forward(&x_in, rope.as_ref(), manifest.position_offset)
        .context("attention forward")?;

    write_f32_bin(&dir.join("y_out.bin"), y_out.as_slice())?;

    let result = PhalanxResult {
        status: "ok".into(),
        shape: manifest.shape,
        y_out: "y_out.bin".into(),
    };
    fs::write(
        dir.join("phalanx_result.json"),
        serde_json::to_string_pretty(&result)? + "\n",
    )?;
    println!("phalanx validate_attention: wrote y_out.bin");
    Ok(())
}

fn read_tensor(path: &Path, shape: &[usize]) -> Result<Tensor> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!(
            "{} length {} not multiple of 4",
            path.display(),
            bytes.len()
        );
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
    Ok(Tensor::from_vec(data, Shape::new(shape.to_vec())?)?)
}

fn write_f32_bin(path: &Path, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
