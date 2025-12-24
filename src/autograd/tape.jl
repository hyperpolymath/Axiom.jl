# Axiom.jl Gradient Tape
#
# Recording-based automatic differentiation.

"""
    GradientTape

Records operations for gradient computation.
"""
mutable struct GradientTape
    operations::Vector{Any}
    is_recording::Bool
    persistent::Bool
end

"""
    GradientTape(; persistent=false)

Create a new gradient tape.

# Arguments
- `persistent`: If true, tape can be used multiple times.
"""
GradientTape(; persistent::Bool=false) = GradientTape([], true, persistent)

"""
    record!(tape, operation)

Record an operation to the tape.
"""
function record!(tape::GradientTape, op)
    if tape.is_recording
        push!(tape.operations, op)
    end
end

"""
    stop_recording!(tape)

Stop recording operations.
"""
function stop_recording!(tape::GradientTape)
    tape.is_recording = false
end

"""
    gradient(tape, target, sources)

Compute gradients of target with respect to sources using recorded tape.
"""
function gradient(tape::GradientTape, target, sources)
    if !tape.persistent
        tape.is_recording = false
    end

    # Replay tape in reverse for backprop
    grads = Dict()
    grads[objectid(target)] = one(eltype(target))

    for op in reverse(tape.operations)
        # Each op stores: (output_id, input_ids, backward_fn)
        out_id, in_ids, backward = op

        if haskey(grads, out_id)
            upstream = grads[out_id]
            downstream = backward(upstream)

            for (id, grad) in zip(in_ids, downstream)
                if haskey(grads, id)
                    grads[id] = grads[id] + grad
                else
                    grads[id] = grad
                end
            end
        end
    end

    [get(grads, objectid(s), nothing) for s in sources]
end

"""
    @with_tape tape expr

Execute expression while recording to tape.
"""
macro with_tape(tape, expr)
    quote
        tape = $(esc(tape))
        tape.is_recording = true
        result = $(esc(expr))
        result
    end
end

# Global default tape
const _default_tape = Ref{Union{GradientTape, Nothing}}(nothing)

"""
    with_gradient_tape(f; persistent=false)

Execute f with a gradient tape and return (result, tape).
"""
function with_gradient_tape(f; persistent::Bool=false)
    tape = GradientTape(persistent=persistent)
    old_tape = _default_tape[]
    _default_tape[] = tape

    try
        result = f()
        return result, tape
    finally
        _default_tape[] = old_tape
    end
end
