# SONNET-TASKS.md — Axiom.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, open-items, and placeholder code in this repo.
> **Honest completion before this file:** ~45-50% (STATE.scm claims 65% — overstated)

---

## GROUND RULES FOR SONNET

1. **Read this entire file before starting any task.**
2. **Do tasks in the order listed.** Earlier tasks unblock later ones.
3. **After completing each task, run the verification command listed for that task.** If it fails, fix it before moving on.
4. **Do NOT mark a task done unless verification passes.**
5. **Update STATE.scm** with honest completion percentages after each task.
6. **Commit after each completed task** with message format: `fix(component): complete <description>`
7. **Julia version:** 1.10+ (check Project.toml compat)
8. **Run full test suite** after every 3 tasks: `cd /var/mnt/eclipse/repos/Axiom.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Fix Autograd — Replace Toy Implementation (HIGHEST PRIORITY)

**Files:** `src/autograd/gradient.jl`, `src/autograd/tape.jl`

**Problem:** The autograd system is ~30% complete. `backward!()` only handles trivial topological sort with no real gradient computation. No support for matrix ops, broadcasting, reshaping. The code itself admits: "This is a minimal implementation - production would use Zygote.jl or Enzyme.jl"

**What to do:**
1. Add `Zygote` to `[deps]` in `Project.toml` (NOT weakdeps — this is core functionality)
2. Rewrite `src/autograd/gradient.jl` to use Zygote as the backend:
   - `gradient(f, params...)` should call `Zygote.gradient(f, params...)`
   - `backward!(tape)` should use Zygote's pullback mechanism
   - `jacobian(f, x)` should use `Zygote.jacobian(f, x)`
3. Keep the `Tape` type in `tape.jl` as a recording/debugging wrapper around Zygote, not a replacement
4. Ensure all existing layer types (`Dense`, `Conv2d`, etc.) work with the new autograd
5. The training loop in `src/training/train.jl` must work end-to-end with the new autograd

**Verification:**
```julia
cd /var/mnt/eclipse/repos/Axiom.jl
julia --project=. -e '
using Axiom
# Test 1: gradient of scalar function
g = Axiom.gradient(x -> x^2 + 3x, 2.0)
@assert g[1] ≈ 7.0 "Scalar gradient failed: got $(g[1]), expected 7.0"

# Test 2: gradient through Dense layer
d = Axiom.Dense(4, 2)
x = randn(Float32, 4)
loss(m) = sum(m(x))
g = Axiom.gradient(loss, d)
@assert g !== nothing "Dense gradient returned nothing"
@assert length(g) > 0 "Dense gradient empty"

# Test 3: jacobian
J = Axiom.jacobian(x -> [x[1]^2, x[1]*x[2]], [3.0, 4.0])
@assert size(J) == (2, 2) "Jacobian wrong size"
println("AUTOGRAD TESTS PASSED")
'
```

---

## TASK 2: Fix Proof Export — Replace Stubs With Real Implementations

**Files:** `src/proof_export.jl`

**Problem:**
- `export_lean()` generates files with `sorry` (unproven placeholder)
- `export_coq()` generates with `Admitted` (unproven placeholder)
- `export_isabelle()` generates with `sledgehammer` comment
- `import_lean_certificate()` → hard `error("not yet implemented")`
- `import_coq_certificate()` → hard `error("not yet implemented")`
- `import_isabelle_certificate()` → hard `error("not yet implemented")`
- Helper functions do naive string replacements, not real translation

**What to do:**
1. **Export functions:** Generate actual proof obligations, not empty templates.
   - `export_lean()`: Generate Lean 4 syntax with `theorem` declarations and actual proof structure from the verification properties. Use `by decide`, `by simp`, `by omega` tactics where applicable for decidable properties (like finite outputs, bounded values). For undecidable properties, leave `sorry` but add a `-- PROOF OBLIGATION: <description>` comment.
   - `export_coq()`: Generate Coq with `Theorem` + `Proof.` blocks. Use `auto`, `omega`, `lia` tactics for arithmetic properties. Mark genuinely unproven obligations with `Admitted (* PROOF OBLIGATION: <description> *)`.
   - `export_isabelle()`: Generate Isabelle/HOL with `lemma` + `proof -` blocks. Use `by auto`, `by simp`, `by arith` where applicable.
2. **Import functions:** Parse proof assistant output files:
   - `import_lean_certificate(path)`: Read a `.lean` file, check for `sorry`-free status (no unproven obligations), extract theorem names and types, return a `ProofCertificate` struct.
   - `import_coq_certificate(path)`: Read a `.v` file, check for `Admitted`-free status, extract theorem names.
   - `import_isabelle_certificate(path)`: Read a `.thy` file, check for `oops`-free status, extract lemma names.
3. **Helper functions:** Replace naive string replacements with proper Julia-to-proof-language type mapping:
   - `Float32` → `real` (Lean), `R` (Coq), `real` (Isabelle)
   - `Vector{Float32}` → `List Real` (Lean), `list R` (Coq), `real list` (Isabelle)
   - Matrix types → appropriate dependent types

**Verification:**
```julia
cd /var/mnt/eclipse/repos/Axiom.jl
julia --project=. -e '
using Axiom

# Create a simple verified model
d = Axiom.Dense(4, 2)

# Test Lean export
lean_code = Axiom.export_lean(d, [:finite_output, :bounded_weights])
@assert occursin("theorem", lean_code) "Lean export missing theorem declarations"
@assert occursin("Real", lean_code) || occursin("real", lean_code) "Lean export missing type mappings"
println("Lean export:\n", lean_code[1:min(200, length(lean_code))])

# Test Coq export
coq_code = Axiom.export_coq(d, [:finite_output])
@assert occursin("Theorem", coq_code) "Coq export missing Theorem"
println("Coq export:\n", coq_code[1:min(200, length(coq_code))])

# Test Isabelle export
isa_code = Axiom.export_isabelle(d, [:finite_output])
@assert occursin("lemma", isa_code) "Isabelle export missing lemma"
println("Isabelle export:\n", isa_code[1:min(200, length(isa_code))])

# Test import (create a mock certificate file)
tmpdir = mktempdir()
lean_cert = joinpath(tmpdir, "test.lean")
write(lean_cert, """
theorem finite_output : ∀ x : Fin 4 → Real, ∃ y : Fin 2 → Real, True := by
  intro x
  exact ⟨fun _ => 0, trivial⟩
""")
cert = Axiom.import_lean_certificate(lean_cert)
@assert cert !== nothing "Import returned nothing"
@assert cert.verified == true || cert.sorry_free == true "Certificate not marked verified"
println("PROOF EXPORT TESTS PASSED")
'
```

---

## TASK 3: Fix HuggingFace Integration — Implement or Remove (DONE)

**STATUS: DONE**
**ACTION: REMOVED**

**REASONING:** The HuggingFace integration was not implemented and the file containing the function stubs has been removed from the repository. Per the task instructions, this feature has been removed.

**Original Description:**
> **Files:** `src/integrations/huggingface.jl`, `src/Axiom.jl` (line ~130 where it's commented out)
>
> **Problem:**
> - Module is **disabled** (commented out in main module)
> - `build_gpt2()`, `build_vit()`, `build_resnet()` → hard `error("not implemented")`
> - `load_weights!()` → empty function body
> - `load_tokenizer()` → returns `nothing`
> - "subtle parsing issue" mentioned but not fixed
>
> **What to do — pick ONE of these approaches:**
>
> **Option A (RECOMMENDED): Implement properly**
> 1. Fix the "subtle parsing issue" in shapes.jl that breaks integration
> 2. Uncomment the `include("integrations/huggingface.jl")` in `src/Axiom.jl`
> 3. Implement `build_gpt2()`: Multi-head attention + feed-forward blocks using existing Dense/LayerNorm layers
> 4. Implement `build_vit()`: Patch embedding + transformer encoder using existing layers
> 5. Implement `build_resnet()`: Conv2d + BatchNorm + residual connections using existing layers
> 6. Implement `load_weights!()`: Parse PyTorch `.bin` files (they're zip files containing numpy arrays — use `ZipFile.jl` + manual binary parsing, or use `PyCall` via the existing weak dependency)
> 7. Implement `load_tokenizer()`: Parse HuggingFace `tokenizer.json` format (JSON with vocab + merges)
>
> **Option B: Remove cleanly**
> If implementation is too complex, remove the file entirely:
> 1. Delete `src/integrations/huggingface.jl`
> 2. Remove all HuggingFace exports from `src/Axiom.jl`
> 3. Remove `PyCall` from `[weakdeps]` in Project.toml
> 4. Remove `AxiomPyTorchExt` from `[extensions]`
> 5. Delete `ext/AxiomPyTorchExt.jl`
> 6. Update README.adoc to remove HuggingFace claims
>
> **Do NOT leave it in its current broken-but-committed state.**
>
> **Verification (Option A):**
> ```julia
> cd /var/mnt/eclipse/repos/Axiom.jl
> julia --project=. -e '
> using Axiom
>
> # Test architecture builders (without weights — just structure)
> gpt2 = Axiom.build_gpt2(; n_layers=2, n_heads=2, d_model=64, vocab_size=100)
> @assert gpt2 !== nothing "GPT-2 builder returned nothing"
>
> vit = Axiom.build_vit(; n_layers=2, n_heads=2, d_model=64, patch_size=16, image_size=224, n_classes=10)
> @assert vit !== nothing "ViT builder returned nothing"
>
> resnet = Axiom.build_resnet(; layers=[2,2,2,2], n_classes=10)
> @assert resnet !== nothing "ResNet builder returned nothing"
>
> println("HUGGINGFACE INTEGRATION TESTS PASSED")
> '
> ```
>
> **Verification (Option B):**
> ```julia
> cd /var/mnt/eclipse/repos/Axiom.jl
> julia --project=. -e '
> using Axiom
> # Verify no broken exports
> @assert !isdefined(Axiom, :load_from_huggingface) "HuggingFace not cleanly removed"
> @assert !isdefined(Axiom, :build_gpt2) "GPT-2 not cleanly removed"
> println("CLEAN REMOVAL VERIFIED")
> '
> ```

---

## TASK 4: Fix GPU Backend Stubs

**Files:** `src/backends/gpu_hooks.jl`, `ext/AxiomCUDAExt.jl`, `ext/AxiomAMDGPUExt.jl`, `ext/AxiomMetalExt.jl`

**Problem:**
- `cuda_available()` returns `nothing` (not even `false`)
- `rocm_available()` returns hardcoded `false`
- `cuda_device_count()` returns hardcoded `0`
- Extension files exist but are minimal

**What to do:**
1. Fix `gpu_hooks.jl` to return proper `false` (not `nothing`) when no GPU package loaded
2. Implement proper detection via extension loading:
   - When CUDA.jl is loaded → `AxiomCUDAExt` activates → `cuda_available()` calls `CUDA.functional()`
   - When AMDGPU.jl is loaded → `AxiomAMDGPUExt` activates → `rocm_available()` calls `AMDGPU.functional()`
   - When Metal.jl is loaded → `AxiomMetalExt` activates → `metal_available()` calls `Metal.functional()`
3. Each extension should implement:
   - `gpu_available()` → `Bool`
   - `gpu_device_count()` → `Int`
   - `to_gpu(tensor)` → GPU tensor
   - `from_gpu(tensor)` → CPU tensor
4. The default (no extension loaded) must return `false`/`0` consistently, never `nothing`

**Verification:**
```julia
cd /var/mnt/eclipse/repos/Axiom.jl
julia --project=. -e '
using Axiom

# Without GPU packages loaded, these should return false/0 (not nothing, not error)
@assert Axiom.cuda_available() === false "cuda_available() should be false, got $(Axiom.cuda_available())"
@assert Axiom.rocm_available() === false "rocm_available() should be false, got $(Axiom.rocm_available())"
@assert Axiom.metal_available() === false "metal_available() should be false, got $(Axiom.metal_available())"
@assert Axiom.cuda_device_count() === 0 "cuda_device_count() should be 0, got $(Axiom.cuda_device_count())"
println("GPU HOOKS TESTS PASSED (no GPU packages loaded)")
'
```

---

## TASK 5: Fix Model Metadata Placeholders

**File:** `src/model_metadata.jl`

**Problem:**
- Line ~212: `verify_and_claim!()` sets `verified = true` without actually verifying anything. Comment says `# open-item: Actually run verification via @prove`
- `input_shape_from_model()` returns `(0,)` placeholder
- `output_shape_from_model()` returns `(0,)` placeholder

**What to do:**
1. `verify_and_claim!()`: Actually call the verification system. If `@prove` macro / verification checker is available, run it. If verification fails, set `verified = false` and include failure reason.
2. `input_shape_from_model(model)`: Inspect model's first layer to determine expected input shape. For `Dense(in, out)` → `(in,)`. For `Conv2d(in_ch, ...)` with known kernel → derive from channel count.
3. `output_shape_from_model(model)`: Inspect model's last layer to determine output shape. For `Dense(in, out)` → `(out,)`.

**Verification:**
```julia
cd /var/mnt/eclipse/repos/Axiom.jl
julia --project=. -e '
using Axiom
d = Axiom.Dense(10, 5)

# Shape inference
in_shape = Axiom.input_shape_from_model(d)
@assert in_shape == (10,) "Expected (10,), got $in_shape"

out_shape = Axiom.output_shape_from_model(d)
@assert out_shape == (5,) "Expected (5,), got $out_shape"

println("MODEL METADATA TESTS PASSED")
'
```

---

## TASK 6: Fix Conv3d Stub

**File:** `src/layers/conv.jl` (line ~94)

**Problem:** `Conv3d` throws `error("Conv3d is conceptually supported but not yet implemented")`

**What to do:**
1. Implement `Conv3d` following the same pattern as `Conv2d` but for 5D tensors (batch, channels, depth, height, width)
2. Support same parameters: kernel_size, stride, padding, dilation
3. Forward pass: 3D convolution using nested loops or `NNlib.conv` if available

**Verification:**
```julia
cd /var/mnt/eclipse/repos/Axiom.jl
julia --project=. -e '
using Axiom
c = Axiom.Conv3d(3, 16; kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
x = randn(Float32, 3, 8, 8, 8, 1)  # channels, D, H, W, batch
y = c(x)
@assert size(y, 1) == 16 "Wrong output channels"
@assert size(y, 5) == 1 "Wrong batch dim"
println("CONV3D TESTS PASSED")
'
```

---

## TASK 7: Fix SMT Extension Dead Code (DONE)

**STATUS: DONE**
**ACTION: FIXED**

**REASONING:** The SMT extension has been fully enabled in `Project.toml`. Redundant/broken Rust FFI code was removed, and parse errors in `SMTLib.jl` were fixed. AST-based block unwrapping was implemented in `prove.jl` to support complex `@prove` blocks.

**Original Description:**
> **File:** `ext/AxiomSMTExt.jl`
>
> **Problem:** Lines ~56-68 have unreachable return statements after an early return at line ~54.


---

## TASK 8: Fix Zig Backend — Either Implement or Remove (DONE)

**STATUS: DONE**
**ACTION: IMPLEMENTED**

**REASONING:** The Zig build system has been updated to 0.15.2, compilation errors fixed, and a full Julia FFI bridge implemented in `src/backends/zig_ffi.jl`. End-to-end verification of matrix multiplication confirmed the backend is functional.

**Original Description:**
> **File:** `src/backends/zig_ffi.jl`, `zig/src/axiom.zig`
>
> **Problem:** All functions throw `ArgumentError` for validation but have no actual `ccall` to any Zig library. The `ZIG_LIB` constant references a path that doesn't exist.


---

## TASK 9: Implement Missing PyTorch/ONNX Exports

**Files:** `src/Axiom.jl` (exports `from_pytorch`, `to_onnx` but functions don't exist)

**Problem:** `from_pytorch()` and `to_onnx()` are exported in the module but the functions are not defined anywhere in the codebase.

**What to do — pick ONE:**

**Option A: Implement**
1. Create `src/interop/pytorch.jl`: `from_pytorch(path::String)` reads a `.pt` or `.pth` file and reconstructs an Axiom model
2. Create `src/interop/onnx.jl`: `to_onnx(model, path::String)` serializes an Axiom model to ONNX format
3. Include both from `src/Axiom.jl`

**Option B (RECOMMENDED): Remove exports**
1. Remove `from_pytorch` and `to_onnx` from the export list in `src/Axiom.jl`
2. Add a doc comment explaining these are planned future features
3. Update README.adoc to remove interop claims

**Do NOT leave phantom exports that error on use.**

**Verification:**
```julia
cd /var/mnt/eclipse/repos/Axiom.jl
julia --project=. -e '
using Axiom
# Check no phantom exports
for sym in [:from_pytorch, :to_onnx]
    if isdefined(Axiom, sym)
        m = getfield(Axiom, sym)
        @assert m isa Function "Export $sym exists but is not a function"
        println("$sym: implemented")
    else
        println("$sym: removed (acceptable)")
    end
end
println("INTEROP EXPORTS CHECK PASSED")
'
```

---

## TASK 10: Update STATE.scm With Honest Numbers

**File:** `.machine_readable/STATE.scm`

**After completing all above tasks**, update STATE.scm with honest completion percentages. The format should reflect what was actually implemented, not aspirational numbers.

**Verification:** Read the file and confirm no component claims >90% unless it truly is >90% complete.

---

## FINAL VERIFICATION — RUN AFTER ALL TASKS

```bash
cd /var/mnt/eclipse/repos/Axiom.jl

# 1. Full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# 2. Check no remaining hard-error stubs
grep -rn 'error(".*not.*implement' src/ ext/ || echo "NO HARD ERROR STUBS REMAINING"

# 3. Check no phantom exports
julia --project=. -e '
using Axiom
for name in names(Axiom)
    try
        getfield(Axiom, name)
    catch e
        println("BROKEN EXPORT: $name — $e")
    end
end
println("ALL EXPORTS VALID")
'

# 4. Check no open-item/fix-item landmines
grep -rn 'open-item\|fix-item\|HACK\|XXX' src/ | head -20
echo "(Some open-items are acceptable for future enhancements, but none should be for core functionality)"
```
