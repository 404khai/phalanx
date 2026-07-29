//! Cross-implementation `SwiGLU` validator binary.
//!
//! Reads a work directory from Odyssey `scripts/validate_swiglu.py`:
//!
//! - `manifest.json` — shapes
//! - `x_in.bin`, `w_gate.bin`, `w_up.bin`, `w_down.bin` — row-major f32 LE
//!
//! Writes `y_out.bin` and `phalanx_result.json`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use phalanx::{Shape, SwiGlu, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Manifest {
    shape: Vec<usize>,
    hidden_size: usize,
    intermediate_size: usize,
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
            eprintln!("validate_swiglu error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: validate_swiglu <work_dir>")?;

    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(dir.join("manifest.json")).context("read manifest.json")?,
    )
    .context("parse manifest.json")?;

    let w_gate = read_tensor(
        &dir.join("w_gate.bin"),
        &[manifest.intermediate_size, manifest.hidden_size],
    )?;
    let w_up = read_tensor(
        &dir.join("w_up.bin"),
        &[manifest.intermediate_size, manifest.hidden_size],
    )?;
    let w_down = read_tensor(
        &dir.join("w_down.bin"),
        &[manifest.hidden_size, manifest.intermediate_size],
    )?;
    let ffn = SwiGlu::from_tensors(w_gate, w_up, w_down).context("build SwiGlu")?;

    let x_in = read_tensor(&dir.join("x_in.bin"), &manifest.shape)?;
    let y_out = ffn.forward(&x_in).context("swiglu forward")?;

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
    println!("phalanx validate_swiglu: wrote y_out.bin");
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
