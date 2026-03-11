<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# Reliability FAQ (Self-Healing, Fault Tolerance, Self-Diagnostics)

This FAQ is a quick reference for the reliability concepts we discussed.

## 1) What is the difference between self-healing, fault tolerance, and self-diagnostics?

- Self-diagnostics: detect and report what is going wrong.
- Fault tolerance: keep service operating correctly when components fail.
- Self-healing: automatically recover from detected faults (retry, failover, fallback, restart) with clear telemetry.

## 2) Is fallback alone enough?

No. Fallback is only one mechanism.

A robust loop is:
- Detect fault.
- Classify fault.
- Contain blast radius.
- Recover (retry/failover/fallback).
- Record diagnostics.
- Alert when patterns repeat.

## 3) Should we return dummy/token values when something breaks?

For critical computation paths: no.

Use:
- Retry or failover to a safe path.
- Explicit error if safe recovery is not possible.

Dummy values are only acceptable for optional, non-critical fields with explicit provenance.

## 4) How do we detect if a backend is constantly faulting?

Track counters over time, then alert on rates/trends.

In Axiom.jl you now have:
- `gpu_runtime_diagnostics()` for runtime counters.
- `gpu_capability_report()` for capability/hook/diagnostic snapshots.
- `scripts/gpu-performance-evidence.jl` for evidence artifacts.

## 5) What counters matter most?

- Compile fallback count.
- Runtime error count.
- Runtime fallback count.
- Recovery success count.
- Consecutive failure streak.
- Error/fallback rate over a rolling window.

## 6) What thresholds are reasonable to start with?

Example starter thresholds:
- Any required backend fallback in CI => fail.
- Fallback rate > 1% over rolling window => alert.
- Consecutive failures >= 5 => circuit-breaker/quarantine backend.
- Performance regression > configured ratio => fail/alert.

Tune thresholds after collecting baseline data.

## 7) Does this need to be done per backend (CUDA/ROCm/Metal/TPU/NPU/DSP/FPGA)?

Yes, per backend family, because failure modes differ.

Reuse the same framework, but keep backend-specific tests, diagnostics, and performance baselines.

## 8) Does CPU/main path need this too?

Yes. CPU path is also production-critical and needs:
- failure-mode tests,
- diagnostics,
- performance baselines,
- regression gates.

## 9) Do APIs (REST/gRPC/GraphQL) and ABI/FFI need similar treatment?

Yes, with subsystem-specific checks:
- API conformance, latency/error budgets, timeout/retry behavior.
- ABI/FFI compatibility matrix, contract tests, smoke and fault-injection coverage.

## 10) What about containers and images for high-security systems?

Use containers, but harden them.

Minimum:
- pinned base image digests,
- non-root runtime,
- read-only filesystem where possible,
- dropped capabilities,
- SBOM + vulnerability scanning,
- signed images/provenance,
- strict runtime/network policy.

Containers are packaging/isolation, not complete security by themselves.

## 11) Does containerization mean Julia cannot work with accelerators?

No. Julia works fine in containers.

The constraint is accelerator runtime access:
- CUDA/ROCm can work with host passthrough and matching drivers.
- Metal is usually best on native macOS runners.

## 12) How do we avoid turning this into endless work?

Use tiered rollout:
- Tier 1: external/production-critical paths.
- Tier 2: high-usage internal paths.
- Tier 3: lower-risk/experimental paths.

Gate by risk and blast radius, not perfection everywhere at once.

## 13) Current Axiom.jl anchors

- Runtime gate: `scripts/readiness-check.sh`
- GPU resilience tests: `test/ci/gpu_resilience.jl`
- GPU fallback tests: `test/ci/gpu_fallback.jl`
- GPU evidence: `scripts/gpu-performance-evidence.jl`
- Coprocessor evidence: `scripts/coprocessor-evidence.jl`
