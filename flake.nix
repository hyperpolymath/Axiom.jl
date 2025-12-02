# SPDX-FileCopyrightText: 2025 Axiom.jl Contributors
# SPDX-License-Identifier: MIT
{
  description = "Axiom.jl - Provably Correct Machine Learning Framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # Julia with required packages
        juliaEnv = pkgs.julia_19;

        # Zig for backend compilation
        zigPkg = pkgs.zig;

        # Development dependencies
        devDeps = with pkgs; [
          # Julia
          juliaEnv

          # Rust toolchain
          rustToolchain
          cargo-watch
          cargo-audit
          cargo-deny

          # Zig
          zigPkg

          # Build tools
          gnumake
          cmake
          pkg-config

          # Testing & Quality
          just
          hyperfine  # Benchmarking
          tokei      # Code statistics

          # Documentation
          mdbook

          # Git & CI
          git
          gh
          act  # Local GitHub Actions

          # Security
          trivy
          grype
        ];

      in
      {
        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = devDeps;

          shellHook = ''
            echo "🔷 Axiom.jl Development Environment"
            echo "   Julia: $(julia --version)"
            echo "   Rust:  $(rustc --version)"
            echo "   Zig:   $(zig version)"
            echo ""
            echo "Available commands:"
            echo "  just --list    # Show all tasks"
            echo "  just build     # Build all backends"
            echo "  just test      # Run tests"
            echo "  just audit     # Security audit"
            echo ""

            # Set up Julia depot
            export JULIA_DEPOT_PATH="$PWD/.julia"
            mkdir -p $JULIA_DEPOT_PATH

            # Rust settings
            export CARGO_HOME="$PWD/.cargo"
            export RUSTFLAGS="-C target-cpu=native"

            # Zig cache
            export ZIG_LOCAL_CACHE_DIR="$PWD/.zig-cache"
          '';

          # Environment variables
          RUST_BACKTRACE = "1";
          RUST_LOG = "info";
        };

        # Packages
        packages = {
          default = self.packages.${system}.axiom-zig;

          # Zig backend library
          axiom-zig = pkgs.stdenv.mkDerivation {
            pname = "axiom-zig";
            version = "0.1.0";
            src = ./zig;

            nativeBuildInputs = [ zigPkg ];

            buildPhase = ''
              zig build -Doptimize=ReleaseFast
            '';

            installPhase = ''
              mkdir -p $out/lib
              cp zig-out/lib/libaxiom_zig.* $out/lib/
            '';
          };

          # Rust backend library
          axiom-rust = pkgs.rustPlatform.buildRustPackage {
            pname = "axiom-rust";
            version = "0.1.0";
            src = ./rust;

            cargoLock = {
              lockFile = ./rust/Cargo.lock;
            };

            buildType = "release";
          };

          # Documentation
          docs = pkgs.stdenv.mkDerivation {
            pname = "axiom-docs";
            version = "0.1.0";
            src = ./docs;

            nativeBuildInputs = [ pkgs.mdbook ];

            buildPhase = ''
              mdbook build
            '';

            installPhase = ''
              mkdir -p $out
              cp -r book/* $out/
            '';
          };
        };

        # Checks (run with `nix flake check`)
        checks = {
          # Rust formatting
          rust-fmt = pkgs.runCommand "rust-fmt-check" {
            buildInputs = [ rustToolchain ];
          } ''
            cd ${./rust}
            cargo fmt --check
            touch $out
          '';

          # Zig formatting
          zig-fmt = pkgs.runCommand "zig-fmt-check" {
            buildInputs = [ zigPkg ];
          } ''
            cd ${./zig}
            zig fmt --check src/
            touch $out
          '';

          # Security audit
          cargo-audit = pkgs.runCommand "cargo-audit" {
            buildInputs = [ pkgs.cargo-audit rustToolchain ];
          } ''
            cd ${./rust}
            cargo audit
            touch $out
          '';
        };

        # Apps (run with `nix run`)
        apps = {
          # Run benchmarks
          bench = flake-utils.lib.mkApp {
            drv = pkgs.writeShellScriptBin "axiom-bench" ''
              cd ${./zig}
              ${zigPkg}/bin/zig build bench
              ./zig-out/bin/bench
            '';
          };

          # Generate documentation
          docs = flake-utils.lib.mkApp {
            drv = pkgs.writeShellScriptBin "axiom-docs" ''
              cd ${./docs}
              ${pkgs.mdbook}/bin/mdbook serve
            '';
          };
        };
      }
    );
}
