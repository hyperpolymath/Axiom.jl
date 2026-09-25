;; SPDX-License-Identifier: MPL-2.0
;; SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
;; guix.scm — GNU Guix package definition for Axiom.jl
;; Usage: guix shell -D -f guix.scm
;;        guix shell -f guix.scm -- julia --project=. -e 'using Pkg; Pkg.test()'
;;        guix build -f guix.scm

(use-modules (guix packages)
             (guix licenses)
             (guix gexp)
             (guix git-download)
             (guix build-system gnu)
             (gnu packages base)
             (gnu packages bash)
             (gnu packages julia)
             (gnu packages rust)
             (gnu packages zig)
             (gnu packages tls)
             (gnu packages pkg-config))

(package
  (name "axiom-jl")
  (version "1.0.0")
  (source (local-file "." "axiom-jl-checkout"
                      #:recursive? #t
                      #:select? (git-predicate ".")))
  (build-system gnu-build-system)
  (arguments
   (list #:tests? #f
         #:phases
         #~(modify-phases %standard-phases
             (delete 'configure)
             (delete 'build)
             (delete 'check)
             (replace 'install
               (lambda _
                 (mkdir-p (string-append #$output "/share/doc/axiom-jl"))
                 (copy-file "README.adoc"
                            (string-append #$output "/share/doc/axiom-jl/README.adoc"))
                 #t)))))
  (inputs (list bash
                coreutils
                julia
                zig
                rust
                openssl
                pkg-config))
  (native-inputs (list pkg-config))
  (synopsis "Axiom.jl — Provably correct machine learning framework")
  (description
   "Axiom.jl is a Julia-native ML framework with compile-time shape verification,
Zig SIMD kernels, Idris2 formal ABI, and hybrid Ed448+Dilithium5 (ML-DSA-87)
certificate signing.  This Guix package provides the reproducible development
environment for Axiom.jl: Julia 1.10+, Zig 0.15+, Rust stable, and OpenSSL 3.x
for the crypto shim.  The Zig shared library (zig/zig-out/lib/libaxiom_zig.so)
and the Rust cdylib (crypto/target/release/libaxiom_crypto.so) are built
via @code{just build-zig} and @code{just build-crypto} inside the shell.")
  (home-page "https://github.com/hyperpolymath/Axiom.jl")
  (license mpl2.0))
