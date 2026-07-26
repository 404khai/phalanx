//! Weight loading: mmap GGUF files, resolve tensor byte spans, materialize dense floats.
//!
//! # Phase 5 scope
//!
//! | Capability | Status |
//! |---|---|
//! | Memory-map GGUF file (`memmap2`) | ✅ |
//! | Quantization block metadata (`block_size` / `type_size`) | ✅ |
//! | Validate tensor payload bounds | ✅ |
//! | Materialize `f32` / `f16` → [`crate::Tensor`] | ✅ |
//! | Dequantize `Q4_K` / `Q8_0` / … | deferred (model kernels) |
//!
//! # Design tradeoff
//!
//! | Approach | Pros | Cons |
//! |---|---|---|
//! | **mmap whole file (chosen)** | No multi-GB copy; OS paging | Requires reviewed `unsafe` for `memmap2` |
//! | Read `tensor_data` into `Vec` | Pure safe Rust | Hosts can't open 70B models |
//! | Per-tensor maps | Fine-grained | Complex offset math; little gain |
//!
//! Dequant kernels wait until Phase 7+ needs them against real layer shapes.

mod error;
mod quant;
mod set;
mod storage;
mod tensor;

pub use error::WeightsError;
pub use quant::QuantMeta;
pub use set::WeightSet;
pub use storage::WeightStorage;
pub use tensor::WeightTensor;
