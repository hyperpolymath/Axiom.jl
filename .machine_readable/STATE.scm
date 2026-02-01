;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Project state tracking for Axiom.jl

(define-module (state axiom)
  #:use-module (ice-9 match)
  #:export (state get-completion-percentage get-blockers get-milestone))

(define state
  '((metadata
      (version . "0.1.0")
      (schema-version . "1.0.0")
      (created . "2025-01-23")
      (updated . "2026-01-28")
      (project . "Axiom.jl")
      (repo . "https://github.com/hyperpolymath/Axiom.jl"))

    (project-context
      (name . "Axiom.jl")
      (tagline . "Provably correct machine learning with compile-time verification")
      (tech-stack . ("Julia" "Rust" "SMT solvers" "Guix" "Nix"))
      (target-platforms . ("Linux" "macOS" "Windows" "BSD")))

    (current-position
      (phase . "core-implementation")
      (overall-completion . 65)
      (components
        ((name . "Type system (Tensor, Shape)")
         (status . "implemented")
         (completion . 90)
         (notes . "Static shapes, dynamic shapes, compile-time verification"))
        ((name . "Core layers (Dense, Conv, Activation)")
         (status . "implemented")
         (completion . 85)
         (notes . "Dense, Conv2d, BatchNorm, LayerNorm, Dropout, Pooling"))
        ((name . "Training infrastructure")
         (status . "implemented")
         (completion . 80)
         (notes . "train! loop, optimizers (SGD, Adam, AdamW), loss functions"))
        ((name . "Automatic differentiation")
         (status . "implemented")
         (completion . 70)
         (notes . "Gradient tracking, backward pass - minimal implementation"))
        ((name . "Data utilities")
         (status . "implemented")
         (completion . 85)
         (notes . "DataLoader, train_test_split, one_hot encoding"))
        ((name . "DSL macros (@axiom, @ensure)")
         (status . "implemented")
         (completion . 80)
         (notes . "@axiom for model definition, @ensure for runtime assertions"))
        ((name . "Verification system")
         (status . "implemented")
         (completion . 75)
         (notes . "Property checking, ValidProbabilities, FiniteOutput"))
        ((name . "Proof certificates")
         (status . "implemented")
         (completion . 80)
         (notes . "Serialization/deserialization, certificate generation"))
        ((name . "Proof assistant export")
         (status . "implemented")
         (completion . 70)
         (notes . "Lean, Coq, Isabelle export - issue #19"))
        ((name . "Rust backend (core ops)")
         (status . "implemented")
         (completion . 60)
         (notes . "2040 LOC: FFI, tensor ops, matmul, conv, activations, pooling"))
        ((name . "Julia backend")
         (status . "implemented")
         (completion . 90)
         (notes . "Pure Julia fallback for all operations"))
        ((name . "Model containers (Sequential, Chain)")
         (status . "implemented")
         (completion . 85)
         (notes . "Pipeline composition, forward pass"))
        ((name . "Model metadata/packaging")
         (status . "implemented")
         (completion . 75)
         (notes . "ModelMetadata, VerificationClaim - issues #15, #16"))
        ((name . "@prove macro (SMT)")
         (status . "partial")
         (completion . 50)
         (notes . "Structure exists, needs SMTLib weak dependency - issue in src/Axiom.jl:36"))
        ((name . "GPU backends (CUDA, ROCm, Metal)")
         (status . "hooks-only")
         (completion . 20)
         (notes . "Interfaces defined, extensions not implemented - issue #12"))
        ((name . "HuggingFace integration")
         (status . "partial")
         (completion . 40)
         (notes . "Hub API, downloading, BERT/RoBERTa partial - TODOs for GPT-2, ViT, ResNet"))
        ((name . "PyTorch model import")
         (status . "stub")
         (completion . 10)
         (notes . "from_pytorch declared, needs .bin parsing - TODO in huggingface.jl:369"))
        ((name . "ONNX export")
         (status . "stub")
         (completion . 5)
         (notes . "to_onnx declared, not implemented"))
        ((name . "Zig backend")
         (status . "skeleton")
         (completion . 5)
         (notes . "zig_ffi.jl exists but minimal"))
        ((name . "Comprehensive test suite")
         (status . "implemented")
         (completion . 80)
         (notes . "291 LOC covering tensors, layers, training, verification"))
        ((name . "Documentation")
         (status . "complete")
         (completion . 85)
         (notes . "README.adoc with examples, features, benchmarks"))
        ((name . "CI/CD")
         (status . "implemented")
         (completion . 75)
         (notes . "GitHub Actions, CodeQL, security checks"))))

      (working-features
        "Compile-time shape verification"
        "Dense and Conv2d layers"
        "SGD, Adam, AdamW optimizers"
        "MSE, cross-entropy loss"
        "DataLoader with batching"
        "@ensure runtime assertions"
        "Property verification (ValidProbabilities, FiniteOutput)"
        "Proof certificate generation"
        "Lean/Coq/Isabelle export"
        "Rust FFI infrastructure"
        "Sequential model composition"
        "Model metadata tracking"
        "Comprehensive test suite (10+ testsets)"))

    (route-to-mvp
      (milestones
        ((name . "Core ML functionality")
         (target-date . "2026-02-10")
         (status . "mostly-complete")
         (items
           "Complete autograd (integrate Zygote.jl or Enzyme.jl)"
           "Add Residual/Skip connections"
           "Implement attention layers (MultiheadAttention)"
           "Add more optimizers (Adagrad, RMSprop complete)"))
        ((name . "SMT verification")
         (target-date . "2026-02-20")
         (status . "in-progress")
         (items
           "Implement @prove macro with SMTLib extension"
           "Add Z3 solver integration"
           "Complete SMT cache (issue in test/runtests.jl:268-283)"
           "Prove softmax sum property"
           "Document verification workflow"))
        ((name . "Backend performance")
         (target-date . "2026-03-01")
         (status . "partially-done")
         (items
           "Complete Rust backend matmul optimization"
           "Build and test Rust shared library"
           "Benchmark Rust vs Julia backends"
           "Add BLAS/LAPACK integration"
           "Document performance comparison"))
        ((name . "GPU acceleration")
         (target-date . "2026-03-15")
         (status . "planned")
         (items
           "Implement CUDA extension (CUDABackend)"
           "Implement Metal extension (MetalBackend) for Apple Silicon"
           "Implement ROCm extension (ROCmBackend) for AMD"
           "Add GPU detection and auto-selection"
           "Benchmark GPU speedups"))
        ((name . "Model import/export")
         (target-date . "2026-03-20")
         (status . "planned")
         (items
           "Complete PyTorch .pth/.bin weight loading"
           "Implement ONNX export (to_onnx)"
           "Complete HuggingFace architecture builders (GPT-2, ViT, ResNet)"
           "Add BERT/RoBERTa full support"
           "Test with pretrained models"))
        ((name . "v0.1.0 Release")
         (target-date . "2026-04-01")
         (status . "planned")
         (items
           "All core layers functional"
           "Verification system working (@ensure, @prove, verify)"
           "Rust backend tested and optimized"
           "Documentation complete"
           "Test coverage >85%"
           "At least one GPU backend working"
           "PyTorch model import working"
           "Example notebooks (MNIST, ImageNet)"))))

    (blockers-and-issues
      (critical
        ())
      (high
        ("@prove macro requires SMTLib weak dependency - move to extension"
         "Autograd is minimal - should integrate Zygote.jl or Enzyme.jl for production"
         "Rust backend not tested end-to-end (needs build + FFI integration test)"))
      (medium
        ("GPU backends are interface-only, need CUDA/Metal/ROCm extensions"
         "PyTorch .bin weight loading needs pickle parser"
         "HuggingFace integration incomplete (GPT-2, ViT, ResNet stubs)"
         "ONNX export not implemented"
         "GitHub issue #24 (SHA) - needs investigation"))
      (low
        ("Zig backend skeleton exists but not prioritized"
         "Model compression/quantization not yet supported"
         "Distributed training not implemented")))

    (critical-next-actions
      (immediate
        "Test Rust backend compilation completes successfully"
        "Add end-to-end model training test (MNIST or simple dataset)"
        "Investigate and resolve GitHub issue #24")
      (this-week
        "Complete @prove macro SMT integration (move to extension)"
        "Integrate Zygote.jl or Enzyme.jl for production autograd"
        "Build Rust shared library and test FFI calls"
        "Document Rust backend setup (AXIOM_RUST_LIB env var)")
      (this-month
        "Implement CUDA extension for GPU acceleration"
        "Complete PyTorch weight loading"
        "Add MNIST training example"
        "Complete HuggingFace BERT architecture"
        "Benchmark Rust vs Julia performance"
        "Increase test coverage to 85%"))

    (session-history
      ((date . "2026-01-28")
       (accomplishments
         "Analyzed full codebase structure (35 Julia files, 2040 LOC Rust)"
         "Identified 20 components with accurate completion percentages"
         "Documented working features and blockers"
         "Created comprehensive STATE.scm based on actual implementation")))))

;; Helper functions
(define (get-completion-percentage)
  "Get overall project completion percentage"
  (assoc-ref (assoc-ref state 'current-position) 'overall-completion))

(define (get-blockers priority)
  "Get blockers by priority (:critical, :high, :medium, :low)"
  (let ((blockers (assoc-ref state 'blockers-and-issues)))
    (assoc-ref blockers priority)))

(define (get-milestone name)
  "Get milestone by name"
  (let* ((route (assoc-ref state 'route-to-mvp))
         (milestones (assoc-ref route 'milestones)))
    (assoc name milestones)))
