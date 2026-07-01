<!--
SPDX-License-Identifier: CC-BY-SA-4.0
SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
-->

# Training & Automatic Differentiation

Optimizers, loss functions, the training loop, automatic differentiation,
and the `@axiom` / `@ensure` / `@prove` declarative DSL.

```@autodocs
Modules = [Axiom]
Public = true
Private = false
Pages = [
    "dsl/axiom_macro.jl",
    "dsl/ensure.jl",
    "dsl/prove.jl",
    "dsl/pipeline.jl",
    "autograd/gradient.jl",
    "autograd/tape.jl",
    "autograd/invertible_rules.jl",
    "training/optimizers.jl",
    "training/loss.jl",
    "training/train.jl",
    "utils/data.jl",
    "utils/initialization.jl",
]
```
