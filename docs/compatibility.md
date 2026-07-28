# Compatibility with Odyssey

**Phalanx Runtime** is the reference inference engine for the **Odyssey** model family.

Normative contract: [`odyssey/spec/`](../../odyssey/spec/README.md) **v1.0.0**.

---

## Support Declaration

```text
supported_odyssey_spec_versions = ["1.0.0"]
```

| Odyssey Spec | Phalanx | Notes |
| --- | --- | --- |
| `1.0.0` | Supported (partial layer coverage) | See [spec-compliance.md](spec-compliance.md) |

---

## Rules

1. **Metadata over inference** — never guess `num_layers`, heads, or RoPE params.
2. **Frozen GGUF names** — [gguf_mapping.md](../../odyssey/spec/gguf_mapping.md).
3. **Math parity** — RoPE / RMSNorm / SwiGLU / attention match Spec equations.
4. **Tokenizer parity** — long-term: identical `encode`/`decode` on `odyssey-bpe` artifacts.
5. **Errors are loud** — missing weights and shape mismatches fail typed.

---

## Version Skew Policy

| Situation | Action |
| --- | --- |
| Model Spec major > runtime support | Refuse load |
| Model Spec major < runtime (deprecated) | Compat mode or refuse per release notes |
| Minor additive features unknown | Ignore only if Spec marks them optional; else refuse |

---

## Related Docs

- [architecture_mapping.md](architecture_mapping.md) — component crosswalk
- [spec-compliance.md](spec-compliance.md) — checklist
- Odyssey [runtime_contract.md](../../odyssey/spec/runtime_contract.md)
