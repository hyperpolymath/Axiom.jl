# SPDX-License-Identifier: MPL-2.0
# Aqua.jl quality-gate testset (G07).
#
# Aqua checks: method ambiguities, unbound type parameters, undefined
# exports, stale/missing deps, [compat] bounds (including `julia`), type
# piracy, and persistent tasks. Axiom passes every one of these with NO
# overrides required -- see the two fixes below that made this possible:
#
# 1. Unbound type parameters (real bug, fixed at the source): `LayerNorm`,
#    `InstanceNorm`, and `GroupNorm` each had a `T` (and `S`) type
#    parameter that only appeared in `Union{_, Nothing}`-typed fields
#    (`γ`/`β` are `nothing` when `affine=false`). Julia's default
#    auto-generated positional constructor for those structs therefore
#    left `T`/`S` unbound from the perspective of `Test.detect_unbound_args`
#    (a `nothing` argument pins no concrete type). Fixed in
#    `src/layers/normalization.jl` by adding an explicit inner constructor
#    per struct that takes `T`/`S` from the `{T}`/`{T, S}` type
#    application itself, so they stay bound even when both fields are
#    `nothing`. `BatchNorm` was never affected: its `running_mean`/
#    `running_var` fields are non-nullable `Vector{T}`, so `T` was always
#    pinned by those.
# 2. Stale dependency (`JSON3`, removed from `Project.toml`): its only
#    consumer, `src/integrations/huggingface.jl`, is not `include()`d from
#    `src/Axiom.jl` (dead/orphaned code, unreachable from the module), so
#    the dependency was genuinely unused by anything Aqua can see.
#
# `Aqua.test_piracies` also passes without a `treat_as_own` override:
# Axiom extends `size`/`sum`/etc. only via methods that dispatch on
# Axiom's OWN `Tensor`/`DynamicTensor` types, which Aqua's own heuristic
# already recognises as ordinary (non-piratic) dispatch.

using Test
using Aqua
using Axiom

@testset "Aqua quality gate" begin
    Aqua.test_all(Axiom)
end
