<!--
SPDX-License-Identifier: CC-BY-SA-4.0
SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
-->

# Verification & Certification

Formal property specification, checking, certificate generation and
serialization, hybrid Ed448+Dilithium5 signing, and proof-assistant export
(Lean/Coq/Isabelle).

```@autodocs
Modules = [Axiom]
Public = true
Private = false
Pages = [
    "verification/properties.jl",
    "verification/checker.jl",
    "verification/certificates.jl",
    "verification/serialization.jl",
    "verification/signing.jl",
    "verification/verification.jl",
    "proof_export.jl",
]
```
