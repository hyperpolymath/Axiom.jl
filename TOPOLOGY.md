# Axiom.jl - System Architecture

```
                         Axiom.jl - Provably Correct Machine Learning
                         ============================================

    User API Layer
    +-----------------------------------------------------------------+
    |  @axiom    @ensure    @prove*    Sequential/Chain/Pipeline      |
    |  Dense  Conv2d  BatchNorm  LayerNorm  MaxPool  Dropout  ReLU   |
    |  train!()    compile()    verify()    from_pytorch()            |
    +-----------------------------------------------------------------+
         |              |              |              |
         v              v              v              v
    +----------+  +-----------+  +-----------+  +-----------+
    |  DSL &   |  | Training  |  | Verifi-   |  | Integra-  |
    |  Macros  |  | Loop      |  | cation    |  | tions     |
    |----------|  |-----------|  |-----------|  |-----------|
    | axiom_   |  | optimiz-  |  | proper-   |  | hugging-  |
    |  macro   |  |  ers.jl   |  |  ties.jl  |  |  face.jl  |
    | ensure   |  | loss.jl   |  | checker   |  | pytorch   |
    | prove*   |  | train.jl  |  | certifi-  |  |  ext      |
    | pipeline |  | gradient* |  |  cates.jl  |  |           |
    +----------+  +-----------+  | serial-   |  +-----------+
                                 |  ize.jl   |
                                 | proof_    |
                                 |  export   |
                                 +-----------+
                                       |
         +-----------------------------+-----------------------------+
         |                             |                             |
         v                             v                             v
    +-----------+               +-----------+               +-----------+
    | SMTLib.jl |               | Lean 4    |               | Coq /     |
    | (bundled) |               | Export*   |               | Isabelle* |
    |-----------|               +-----------+               +-----------+
    | z3, cvc5  |
    | yices,    |
    | mathsat   |
    +-----------+

    Backend Abstraction Layer
    +-----------------------------------------------------------------+
    |  abstract.jl  -  AbstractBackend / set_backend! / compile()     |
    |  CompilationTarget / MixedPrecisionWrapper / optimize_model*    |
    +-----------------------------------------------------------------+
         |              |              |              |
         v              v              v              v
    +-----------+  +-----------+  +-----------+  +-----------+
    | Julia     |  | Rust      |  | Zig       |  | GPU       |
    | Backend   |  | Backend   |  | Backend   |  | Backends  |
    |-----------|  |-----------|  |-----------|  |-----------|
    | julia_    |  | rust_     |  | zig_      |  | gpu_      |
    |  backend  |  |  ffi.jl   |  |  ffi.jl   |  |  hooks.jl |
    |  .jl      |  |   |       |  |   |       |  |   |       |
    | (default) |  |   v       |  |   v       |  |   v       |
    |           |  | rust/     |  | zig/      |  | ext/      |
    +-----------+  | src/      |  | src/      |  | CUDA      |
                   | ffi.rs    |  | axiom.zig |  | AMDGPU    |
                   | ops/      |  | matmul    |  | Metal     |
                   |  matmul   |  | activa-   |  | PyTorch   |
                   |  activ    |  |  tions    |  +-----------+
                   |  conv     |  | conv      |
                   |  pool     |  | pool      |
                   |  norm     |  | norm      |
                   +-----------+  | attention |
                                  +-----------+

    * = disabled, stub, or placeholder
```

## Completion Dashboard

### Core Framework
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| Type System (Tensor)   | Done   | `██████████` 100%              |
| Layer Definitions      | Done   | `██████████` 100%              |
| Activation Functions   | Done   | `██████████` 100%              |
| Optimizers (SGD/Adam)  | Done   | `██████████` 100%              |
| Loss Functions         | Done   | `██████████` 100%              |
| Training Loop          | Partial| `██████░░░░` 60%               |
| Autograd               | Stub   | `██░░░░░░░░` 20%               |
| Data Utilities         | Done   | `██████████` 100%              |
| Model Containers       | Done   | `██████████` 100%              |

### DSL & Macros
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| @axiom macro           | Done   | `██████████` 100%              |
| @ensure macro          | Done   | `██████████` 100%              |
| @prove macro           | Disabled| `██████░░░░` 60%              |
| Pipeline DSL           | Done   | `██████████` 100%              |

### Verification System
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| Property Checking      | Done   | `██████████` 100%              |
| Proof Certificates     | Done   | `██████████` 100%              |
| Serialization          | Done   | `██████████` 100%              |
| SMTLib.jl (bundled)    | Done   | `████████░░` 80%               |
| Lean 4 Export          | Stub   | `██░░░░░░░░` 20%               |
| Coq Export             | Stub   | `██░░░░░░░░` 20%               |
| Isabelle Export        | Stub   | `██░░░░░░░░` 20%               |
| Proof Import           | None   | `░░░░░░░░░░` 0%                |

### Backends - Julia (Reference)
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| matmul                 | Done   | `██████████` 100%              |
| conv2d                 | Done   | `██████████` 100%              |
| activations (all)      | Done   | `██████████` 100%              |
| batchnorm / layernorm  | Done   | `██████████` 100%              |
| pooling (max/avg/glob) | Done   | `██████████` 100%              |
| dropout / flatten      | Done   | `██████████` 100%              |

### Backends - Rust (rust/ 2040 LOC)
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| FFI exports (ffi.rs)   | Done   | `██████████` 100%              |
| matmul (tiled+Rayon)   | Done   | `██████████` 100%              |
| activations (12 funcs) | Done   | `██████████` 100%              |
| conv2d                 | Done   | `██████████` 100%              |
| pooling (max/glob_avg) | Done   | `██████████` 100%              |
| norm (batch/layer/rms) | Done   | `██████████` 100%              |
| SMT runner             | Done   | `██████████` 100%              |
| Julia-side ccall wiring| Done   | `████████░░` 80%               |
| End-to-end dispatch    | Broken | `████░░░░░░` 40%               |

### Backends - Zig (zig/ 2275 LOC)
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| matmul (SIMD tiled)    | Done   | `██████████` 100%              |
| activations (8 funcs)  | Done   | `██████████` 100%              |
| conv2d + depthwise     | Done   | `██████████` 100%              |
| pooling (max/avg/glob) | Done   | `██████████` 100%              |
| norm (batch/layer/rms) | Done   | `██████████` 100%              |
| flash attention        | Done   | `██████████` 100%              |
| rotary embeddings      | Done   | `██████████` 100%              |
| Julia-side ccall wiring| Done   | `██████████` 100%              |
| Compiled .so artifact  | None   | `░░░░░░░░░░` 0%                |

### Backends - GPU Extensions
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| CUDA: matmul           | Done   | `██████████` 100%              |
| CUDA: relu             | Done   | `██████████` 100%              |
| CUDA: softmax          | Done   | `██████████` 100%              |
| CUDA: conv2d           | None   | `░░░░░░░░░░` 0%                |
| CUDA: batchnorm        | None   | `░░░░░░░░░░` 0%                |
| CUDA: pooling          | None   | `░░░░░░░░░░` 0%                |
| ROCm: matmul/relu/soft | Done   | `██████████` 100%              |
| ROCm: conv2d/norm/pool | None   | `░░░░░░░░░░` 0%                |
| Metal: matmul/relu/soft| Done   | `██████████` 100%              |
| Metal: conv2d/norm/pool| None   | `░░░░░░░░░░` 0%                |

### Integrations
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| PyTorch import/export  | Done   | `████████░░` 80%               |
| HuggingFace framework  | Partial| `████░░░░░░` 40%               |
| HF model converters    | Stub   | `█░░░░░░░░░` 10%               |
| Model Metadata         | Partial| `██████░░░░` 60%               |

### Compile & Optimization
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| Backend dispatch       | Done   | `██████████` 100%              |
| Mixed precision        | Partial| `██████░░░░` 60%               |
| fold_batchnorm         | Stub   | `░░░░░░░░░░` 0%                |
| fold_constants         | Stub   | `░░░░░░░░░░` 0%                |
| dead code elimination  | Stub   | `░░░░░░░░░░` 0%                |

### Infrastructure
| Component              | Status | Progress                       |
|------------------------|--------|--------------------------------|
| CI/CD (GitHub Actions) | Partial| `██████░░░░` 60%               |
| Tests (~46-49 passing) | Good   | `████████░░` 80%               |
| Benchmarks             | Done   | `██████████` 100%              |
| Documentation (wiki)   | Done   | `████████░░` 80%               |
| RSR Compliance         | Poor   | `██░░░░░░░░` 20%               |

## Key Dependencies

| Dependency     | Purpose                    | Required |
|----------------|----------------------------|----------|
| Julia >= 1.9   | Runtime                    | Yes      |
| LinearAlgebra  | Matrix operations          | Yes      |
| SHA            | Proof certificate hashing  | Yes      |
| JSON           | Metadata serialization     | Yes      |
| CUDA.jl        | NVIDIA GPU acceleration    | Optional |
| AMDGPU.jl      | AMD GPU acceleration       | Optional |
| Metal.jl       | Apple GPU acceleration     | Optional |
| PyCall.jl      | PyTorch interop            | Optional |
| Rust toolchain | Rust backend compilation   | Optional |
| Zig toolchain  | Zig backend compilation    | Optional |
| Z3/CVC5        | SMT solver for @prove      | Optional |

## Overall: ~55% complete

**Strongest areas:** Core layers, activations, Rust/Zig kernel implementations, SMTLib
**Weakest areas:** Autograd (placeholder), GPU ext coverage, proof assistant integration, compile optimizations, RSR compliance
