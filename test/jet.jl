# SPDX-License-Identifier: MPL-2.0
# JET.jl static-analysis testset (G07).
#
# `JET.report_package`/`test_package` walk the whole dependency graph by
# default: an unscoped `report_package(Axiom)` currently reports 66
# "possible errors", essentially all of them rooted in transitive deps
# Axiom does not control (JSON/StructUtils conversion fallbacks,
# LinearAlgebra QR internals, Zygote/ZygoteRules adjoint kwcall plumbing,
# Base.Broadcast internals, ...). We scope the report to Axiom's own code
# via JET's native `target_modules` config (see `JET.configured_reports` /
# `JET.LastFrameModule` in JETBase.jl): this is a documented, JET-native
# scoping mechanism, not a hand-rolled file-path filter.
#
# Verified empirically (not just by reading JET's source) that this
# scoping is not blanket-hiding genuine Axiom-side bugs: with
# `target_modules = (Axiom,)`, `report_package(Axiom)` returns exactly
# ZERO reports, including the two candidates that looked at first glance
# like they might be Axiom-origin:
#   - `transform_pipeline`/`is_pipeline_expr` in src/dsl/axiom_macro.jl:
#     JET's unscoped run flags a `Symbol has no field args` error whose
#     OWN innermost frame is `Base.getproperty`/`Base.getfield`, reached
#     because `is_pipeline_expr(expr::Union{Expr,Symbol})` can't prove to
#     JET that the `expr isa Expr &&` short-circuit in the caller already
#     excludes `Symbol` before `.args` is accessed -- unreachable at
#     runtime, and its root cause is a Base call, not an Axiom one.
#   - `prove_property` in src/dsl/prove.jl calling `Axiom.smt_proof(...)`:
#     `smt_proof` is an extension stub (`function smt_proof end`) only
#     implemented by `AxiomSMTExt` when the `SMTLib` weakdep is loaded;
#     `report_package` analyzes bare `Axiom` without loading extensions,
#     so it structurally can't see that method. The call site itself is
#     guarded by `Base.get_extension(Axiom, :AxiomSMTExt) !== nothing`,
#     so it is never reached unless the method exists.
# Any *new* Axiom-origin JET finding will still fail this test -- this
# comment documents why the count is 0 today, not an allowlist of
# suppressed findings.

using Test
using JET
using Axiom

@testset "JET static analysis gate" begin
    JET.test_package(Axiom; target_modules = (Axiom,), toplevel_logger = nothing)
end
