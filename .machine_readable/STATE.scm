;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Current project state for Axiom.jl

(define project-state
  `((metadata
      ((version . "1.0.0")
       (schema-version . "1")
       (created . "2026-01-10T13:47:48+00:00")
       (updated . "2026-02-20T00:00:00+00:00")
       (project . "Axiom.jl")
       (repo . "Axiom.jl")))

    (current-position
      ((phase . "Active Development")
       (overall-completion . 55)
       (working-features
         ("tensor-types" "dense-layer" "conv2d-layer" "activations"
          "normalization-layers" "pooling-layers" "sequential-pipeline"
          "optimizers-sgd-adam-adamw-rmsprop" "loss-functions"
          "training-loop-finite-diff" "ensure-macro" "axiom-macro"
          "property-checking" "proof-certificates" "proof-serialization"
          "smtlib-solver-interface" "julia-backend-all-ops"
          "rust-ffi-all-ops" "zig-ffi-all-ops"
          "gpu-hooks-matmul-relu-softmax"
          "pytorch-import-export" "model-metadata"
          "proof-export-lean-coq-isabelle-stubs"
          "data-loader" "benchmarks"))))

    (route-to-mvp
      ((milestones
        ((v0.1.0 . ((items . ("Core layers" "Julia backend" "Basic verification"
                               "Training loop" "Rust FFI" "Zig FFI"
                               "GPU extension hooks" "SMTLib integration"
                               "PyTorch interop" "Proof certificates"))
                    (status . "mostly-complete")))
         (v0.2.0 . ((items . ("Enable @prove macro" "Real autograd via Zygote"
                               "Complete GPU extensions" "Rust end-to-end dispatch"
                               "Build zig .so artifact" "Complete HuggingFace converters"
                               "Compile optimizations"))
                    (status . "not-started")))
         (v1.0.0 . ((items . ("Proof assistant import" "Real proof translation"
                               "Production model save/load" "Full HuggingFace model zoo"
                               "Performance parity benchmarks" "Security audit"
                               "RSR compliance"))
                    (status . "not-started")))))))

    (blockers-and-issues
      ((critical
         (("@prove macro disabled" .
           "SMTLib weak dependency import ordering prevents @prove from loading in main module. Needs extension refactor.")
          ("Autograd is placeholder" .
           "compute_gradients uses finite differences. Needs Zygote.jl or Enzyme.jl integration for production training.")))
       (high
         (("Rust end-to-end dispatch broken" .
           "rust_forward(Dense) always falls back to Julia even when Rust lib loaded and symbol resolved.")
          ("Zig library not compiled" .
           "No .so artifact committed. Users must run zig build manually.")
          ("GPU extensions incomplete" .
           "CUDA/ROCm/Metal only implement matmul, relu, softmax. Missing conv2d, batchnorm, pooling.")
          ("Git author wrong" .
           "26/28 commits authored as 'Your Name <you@example.com>'. Needs rebase or note.")))
       (medium
         (("HuggingFace converters unimplemented" .
           "GPT-2, ViT, ResNet conversion all error(). BERT partially attempted.")
          ("Compile optimizations are no-ops" .
           "fold_batchnorm, fold_constants, eliminate_dead_code all return model unchanged.")
          ("Proof export generates stubs" .
           "Lean/Coq/Isabelle export produces sorry/Admitted, not real proofs.")
          ("Proof import errors" .
           "import_lean_certificate, import_coq_certificate all call error().")
          ("Model save/load broken" .
           "save_model uses repr(), load_model! is a no-op. Needs JLD2/BSON.")))
       (low
         (("SPDX header inconsistency" .
           "Main module says MIT, other files say PMPL-1.0-or-later.")
          ("Justfile is Nix-only" .
           "All recipes require nix but no flake.nix exists in repo.")
          ("Missing RSR workflows" .
           "Only 11/17 required workflows present.")
          ("Repo in wrong location" .
           "Currently at /home/hyper/repo-overwrite/ not canonical ~/Documents/hyperpolymath-repos/")))))

    (critical-next-actions
      ((immediate
         ("Fix @prove macro - refactor SMTLib as proper package extension"
          "Wire Rust backend end-to-end for Dense layer forward pass"
          "Add zig build step and commit build.zig properly"
          "Move repo to canonical location"))
       (this-week
         ("Integrate Zygote.jl for real autograd"
          "Complete GPU extensions (conv2d, batchnorm, pooling)"
          "Fix SPDX headers to PMPL-1.0-or-later"
          "Replace Nix Justfile with Julia-native recipes"))
       (this-month
         ("Implement HuggingFace model converters"
          "Add compile optimization passes"
          "Real proof translation (not sorry stubs)"
          "Production model save/load via JLD2"
          "RSR compliance (workflows, AI manifest)"))))

    (session-history
      ((session-2026-02-20
         ((actions . ("Full codebase audit" "Created TOPOLOGY.md"
                      "Moved SCM files to .machine_readable/"
                      "Updated STATE.scm with accurate status"
                      "Rewrote ECOSYSTEM.scm (was corrupted with badge URL)"
                      "Updated META.scm with architecture decisions"
                      "Replaced Nix Justfile with Julia-native recipes"))
          (agent . "claude-opus-4-6")))))))
