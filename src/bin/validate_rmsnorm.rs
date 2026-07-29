//! Cross-implementation `RMSNorm` validator binary.
//!
//! Reads a work directory produced by Odyssey `scripts/validate_rmsnorm.py`:
//!
//! - `manifest.json` — shape + `eps` + seed metadata
//! - `x_in.bin` / `gamma.bin` — row-major f32 little-endian
//!
//! Writes `y_out.bin` and `phalanx_result.json`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use phalanx::{RmsNorm, Shape, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Manifest {
    shape: Vec<usize>,
    hidden_size: usize,
    eps: f32,
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
            eprintln!("validate_rmsnorm error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: validate_rmsnorm <work_dir>")?;

    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(dir.join("manifest.json")).context("read manifest.json")?,
    )
    .context("parse manifest.json")?;

    let gamma = read_tensor(&dir.join("gamma.bin"), &[manifest.hidden_size])?;
    let norm = RmsNorm::from_tensor(gamma, manifest.eps).context("build RmsNorm")?;

    let x_in = read_tensor(&dir.join("x_in.bin"), &manifest.shape)?;
    let y_out = norm.forward(&x_in).context("rmsnorm forward")?;

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
    println!("phalanx validate_rmsnorm: wrote y_out.bin");
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
