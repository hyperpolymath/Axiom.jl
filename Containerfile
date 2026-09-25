# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
# Containerfile — sealed-container escape hatch for Axiom.jl
# Estate policy (3-practice/LANGUAGE-POLICY.adoc): Guix is primary,
# sealed container is the escape hatch for not-in-Guix / non-free tail.
# This file is Podman-verifiable where Guix is not installable and satisfies
# check-package-policy.sh as the escape hatch (requires at least one RUN).
#
# Base: Chainguard Wolfi (per estate container policy: cgr.dev/chainguard, not debian/ubuntu)
# Build: podman build -f Containerfile -t axiom-jl:latest .
# Run:   podman run --rm -it axiom-jl:latest julia --project=. -e 'using Axiom; println(Axiom.VERSION)'

FROM cgr.dev/chainguard/wolfi-base:latest AS base

# Install Julia, Zig, Rust, and system deps via Wolfi apk
RUN apk add --no-cache \
    bash \
    coreutils \
    git \
    julia \
    zig \
    rust \
    cargo \
    openssl-dev \
    pkgconf \
    build-base \
    just

WORKDIR /app

# Copy source
COPY . .

# Build Zig backend and Rust crypto shim (best-effort in container build;
# failures do not block the image — runtime `just` can rebuild)
RUN zig build -Doptimize=ReleaseFast || echo "Zig build skipped (no zig or no source)"
RUN cargo build --release --manifest-path crypto/Cargo.toml || echo "Crypto build skipped"

# Default command: print Axiom.jl version and available backends
CMD ["julia", "--project=.", "-e", "using Axiom; println(\"Axiom.jl \", Axiom.VERSION, \" container ready\")"]

# Expose no network port by default; model serving ports are runtime-configurable:
#   podman run -p 8080:8080 axiom-jl:latest julia --project=. -e 'using Axiom; serve_rest(model; port=8080)'
EXPOSE 8080
