;; SPDX-License-Identifier: MPL-2.0
;; SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
;; manifest.scm — Guix manifest for Axiom.jl development environment
;; Usage: guix shell -m manifest.scm
;;        guix shell -m manifest.scm -- julia --project=. -e 'using Pkg; Pkg.test()'
;; Alternative to guix.scm; both satisfy the estate Guix primary policy.
;; guix.scm is a full package definition; this manifest is a lightweight dev shell.

(specifications->manifest
 '("bash"
   "coreutils"
   "git"
   "just"
   "julia"
   "zig"
   "rust"
   "openssl"
   "pkg-config"
   "zlib"))
