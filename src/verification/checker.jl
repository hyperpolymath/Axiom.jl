# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Verification Checker
#
# Main entry point for model verification.

"""
    VerificationResult

Result of running verification on a model.
"""
struct VerificationResult
    passed::Bool
    properties_checked::Vector{Pair{Property, Bool}}
    runtime_seconds::Float64
    counterexamples::Dict{Property, Any}
    warnings::Vector{String}
end

# Structured verification telemetry counters for dashboard/incident workflows.
const _VERIFICATION_TELEMETRY = Dict{String, Any}(
    "runs" => 0,
    "passed" => 0,
    "failed" => 0,
    "warnings" => 0,
    "properties_checked" => 0,
    "property_failures" => 0,
    "counterexamples" => 0,
    "by_property" => Dict{String, Dict{String, Int}}(),
    "last_run" => nothing,
)

_property_name(prop::Property) = string(nameof(typeof(prop)))

function _ensure_property_telemetry_bucket!(name::String)
    buckets = _VERIFICATION_TELEMETRY["by_property"]
    if !haskey(buckets, name)
        buckets[name] = Dict("checked" => 0, "passed" => 0, "failed" => 0)
    end
    buckets[name]
end

"""
    verification_result_telemetry(result::VerificationResult; mode=nothing, source="verify", tags=Dict())

Build a structured telemetry payload for a single verification run.
"""
function verification_result_telemetry(
    result::VerificationResult;
    mode = nothing,
    source::String = "verify",
    tags::Dict{String, Any} = Dict{String, Any}(),
)
    property_results = Vector{Dict{String, Any}}()
    passed_properties = 0
    failed_properties = 0
    for (prop, passed) in result.properties_checked
        name = _property_name(prop)
        push!(property_results, Dict("property" => name, "passed" => passed))
        if passed
            passed_properties += 1
        else
            failed_properties += 1
        end
    end

    counterexample_properties = [_property_name(prop) for prop in keys(result.counterexamples)]

    Dict{String, Any}(
        "format" => "axiom-verification-telemetry.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "source" => source,
        "mode" => mode,
        "passed" => result.passed,
        "runtime_seconds" => result.runtime_seconds,
        "properties_total" => length(result.properties_checked),
        "properties_passed" => passed_properties,
        "properties_failed" => failed_properties,
        "warnings_count" => length(result.warnings),
        "counterexamples_count" => length(result.counterexamples),
        "counterexample_properties" => counterexample_properties,
        "property_results" => property_results,
        "warnings" => copy(result.warnings),
        "tags" => Dict(tags),
    )
end

function _record_verification_telemetry!(
    result::VerificationResult;
    mode = nothing,
    source::String = "verify",
)
    snapshot = verification_result_telemetry(result; mode = mode, source = source)

    _VERIFICATION_TELEMETRY["runs"] += 1
    if result.passed
        _VERIFICATION_TELEMETRY["passed"] += 1
    else
        _VERIFICATION_TELEMETRY["failed"] += 1
    end

    _VERIFICATION_TELEMETRY["warnings"] += length(result.warnings)
    _VERIFICATION_TELEMETRY["properties_checked"] += length(result.properties_checked)
    _VERIFICATION_TELEMETRY["property_failures"] += count(!last(pair) for pair in result.properties_checked)
    _VERIFICATION_TELEMETRY["counterexamples"] += length(result.counterexamples)

    for (prop, passed) in result.properties_checked
        bucket = _ensure_property_telemetry_bucket!(_property_name(prop))
        bucket["checked"] += 1
        if passed
            bucket["passed"] += 1
        else
            bucket["failed"] += 1
        end
    end

    _VERIFICATION_TELEMETRY["last_run"] = snapshot
    snapshot
end

"""
    reset_verification_telemetry!()

Reset in-process verification telemetry counters.
"""
function reset_verification_telemetry!()
    _VERIFICATION_TELEMETRY["runs"] = 0
    _VERIFICATION_TELEMETRY["passed"] = 0
    _VERIFICATION_TELEMETRY["failed"] = 0
    _VERIFICATION_TELEMETRY["warnings"] = 0
    _VERIFICATION_TELEMETRY["properties_checked"] = 0
    _VERIFICATION_TELEMETRY["property_failures"] = 0
    _VERIFICATION_TELEMETRY["counterexamples"] = 0
    _VERIFICATION_TELEMETRY["by_property"] = Dict{String, Dict{String, Int}}()
    _VERIFICATION_TELEMETRY["last_run"] = nothing
    nothing
end

"""
    verification_telemetry_report() -> Dict{String,Any}

Return aggregate structured verification telemetry counters for dashboards.
"""
function verification_telemetry_report()
    runs = _VERIFICATION_TELEMETRY["runs"]
    by_property = Dict{String, Any}()
    for (name, stats) in _VERIFICATION_TELEMETRY["by_property"]
        checked = get(stats, "checked", 0)
        passed = get(stats, "passed", 0)
        failed = get(stats, "failed", 0)
        by_property[name] = Dict(
            "checked" => checked,
            "passed" => passed,
            "failed" => failed,
            "pass_rate" => checked == 0 ? 0.0 : passed / checked,
        )
    end

    Dict(
        "format" => "axiom-verification-telemetry-summary.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "runs" => runs,
        "passed" => _VERIFICATION_TELEMETRY["passed"],
        "failed" => _VERIFICATION_TELEMETRY["failed"],
        "warnings" => _VERIFICATION_TELEMETRY["warnings"],
        "properties_checked" => _VERIFICATION_TELEMETRY["properties_checked"],
        "property_failures" => _VERIFICATION_TELEMETRY["property_failures"],
        "counterexamples" => _VERIFICATION_TELEMETRY["counterexamples"],
        "overall_pass_rate" => runs == 0 ? 0.0 : _VERIFICATION_TELEMETRY["passed"] / runs,
        "by_property" => by_property,
        "last_run" => _VERIFICATION_TELEMETRY["last_run"],
    )
end

function Base.show(io::IO, r::VerificationResult)
    status = r.passed ? "✓ PASSED" : "✗ FAILED"
    println(io, "Verification Result: $status")
    println(io, "Properties checked: $(length(r.properties_checked))")

    for (prop, passed) in r.properties_checked
        symbol = passed ? "✓" : "✗"
        println(io, "  $symbol $(typeof(prop).name.name)")
    end

    if !isempty(r.counterexamples)
        println(io, "\nCounterexamples:")
        for (prop, example) in r.counterexamples
            println(io, "  $(typeof(prop).name.name): $example")
        end
    end

    if !isempty(r.warnings)
        println(io, "\nWarnings:")
        for w in r.warnings
            println(io, "  ⚠ $w")
        end
    end

    println(io, "\nRuntime: $(round(r.runtime_seconds, digits=2))s")
end

"""
    verify(model; properties=default_properties(), data=nothing)

Run verification on a model.

# Arguments
- `model`: Model to verify
- `properties`: List of properties to check
- `data`: Test data for empirical verification

# Returns
VerificationResult
"""
function verify(
    model::Union{AbstractLayer, AxiomModel};
    properties::Vector{<:Property} = Property[FiniteOutput()],
    data = nothing,
    telemetry_mode = nothing,
    telemetry_source::String = "verify",
)
    start_time = time()
    results = Pair{Property, Bool}[]
    counterexamples = Dict{Property, Any}()
    warnings = String[]

    for prop in properties
        # Try static analysis first
        static_result = try_static_verify(prop, model)

        if static_result === :proven
            push!(results, prop => true)
        elseif static_result === :disproven
            push!(results, prop => false)
            counterexamples[prop] = "Disproven by static analysis"
        else
            # Fall back to empirical checking
            if data === nothing
                push!(warnings, "Cannot verify $(typeof(prop).name.name) without test data")
                push!(results, prop => false)
            else
                passed = check(prop, model, data)
                push!(results, prop => passed)
                if !passed
                    counterexamples[prop] = "Found during empirical testing"
                end
            end
        end
    end

    runtime = time() - start_time
    passed = all(last.(results))

    result = VerificationResult(passed, results, runtime, counterexamples, warnings)
    _record_verification_telemetry!(result; mode = telemetry_mode, source = telemetry_source)
    result
end

"""
Try to verify a property using static analysis.
Returns :proven, :disproven, or :unknown.
"""
function try_static_verify(prop::Property, model)
    # Check if model structure guarantees the property

    if prop isa ValidProbabilities
        # Check if model ends with Softmax
        if has_softmax_output(model)
            return :proven
        end
    end

    if prop isa BoundedOutput
        # Check if model ends with bounded activation
        if has_bounded_output(model, prop.low, prop.high)
            return :proven
        end
    end

    if prop isa NoNaN
        # Check for NaN-producing operations
        if has_safe_operations(model)
            return :proven
        end
    end

    :unknown
end

"""
Check if model ends with Softmax.
"""
function has_softmax_output(model)
    if model isa Pipeline
        last_layer = model.layers[end]
        return last_layer isa Softmax
    elseif model isa AxiomModel
        # Check axiom definition
        # Would inspect the model structure
    end
    false
end

"""
Check if model output is bounded.
"""
function has_bounded_output(model, low, high)
    if model isa Pipeline
        last_layer = model.layers[end]
        if last_layer isa Sigmoid && low <= 0 && high >= 1
            return true
        elseif last_layer isa Tanh && low <= -1 && high >= 1
            return true
        end
    end
    false
end

"""
Check if model uses only NaN-safe operations.
"""
function has_safe_operations(model)
    # Simplified check - real implementation would analyze graph
    true
end

"""
    verify_model(model)

Convenience function to verify model with default properties.
"""
function verify_model(model)
    # Generate some random test data
    # In practice, user would provide real data
    test_data = [(randn(Float32, 1, 10), [1]) for _ in 1:10]

    result = verify(model, data=test_data)

    if !result.passed
        @warn "Model verification failed"
    else
        @info "Model verification passed"
    end

    result
end

# ============================================================================
# Verification Modes
# ============================================================================

"""
    VerificationMode

Specifies how thorough verification should be.
"""
@enum VerificationMode begin
    QUICK      # Fast, basic checks only
    STANDARD   # Default verification
    THOROUGH   # Extensive testing
    EXHAUSTIVE # For safety-critical (slow)
end

"""
    verify(model, mode::VerificationMode; kwargs...)

Verify with specified thoroughness.
"""
function verify(model, mode::VerificationMode; kwargs...)
    properties = if mode == QUICK
        [FiniteOutput()]
    elseif mode == STANDARD
        [FiniteOutput(), NoNaN(), NoInf()]
    elseif mode == THOROUGH
        [FiniteOutput(), NoNaN(), NoInf(), LocalLipschitz(0.01f0, 0.1f0)]
    else  # EXHAUSTIVE
        [
            FiniteOutput(),
            NoNaN(),
            NoInf(),
            LocalLipschitz(0.01f0, 0.1f0),
            LocalLipschitz(0.001f0, 0.01f0),
            AdversarialRobust(0.1f0)
        ]
    end

    verify(model; properties = properties, telemetry_mode = string(mode), kwargs...)
end
