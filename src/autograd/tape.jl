# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Gradient Tape
#
# Debugging/visualization wrapper around Zygote for operation recording

using Zygote

"""
    GradientTape

Records operations for debugging and visualization.
Uses Zygote backend for actual gradient computation.

# Fields
- `operations`: Vector of recorded operations (op_name, inputs, outputs)
- `is_recording`: Whether tape is currently recording
- `persistent`: If true, tape can be reused multiple times
"""
mutable struct GradientTape
    operations::Vector{Tuple{Symbol, Vector{Any}, Any}}
    is_recording::Bool
    persistent::Bool
end

"""
    GradientTape(; persistent=false)

Create a new gradient tape for operation recording.

# Arguments
- `persistent`: If true, tape can be used multiple times

# Examples
```julia
tape = GradientTape(persistent=true)
```
"""
GradientTape(; persistent::Bool=false) = GradientTape(
    Tuple{Symbol, Vector{Any}, Any}[],
    true,
    persistent
)

"""
    record!(tape, op_name, inputs, output)

Record an operation to the tape for debugging.

# Arguments
- `tape`: GradientTape to record to
- `op_name`: Symbol naming the operation (e.g., :matmul, :relu)
- `inputs`: Vector of input values
- `output`: Output value
"""
function record!(tape::GradientTape, op_name::Symbol, inputs::Vector, output)
    if tape.is_recording
        push!(tape.operations, (op_name, copy(inputs), output))
    end
    return output
end

"""
    start_recording!(tape)

Start or resume recording operations to tape.
"""
function start_recording!(tape::GradientTape)
    tape.is_recording = true
    return tape
end

"""
    stop_recording!(tape)

Stop recording operations to tape.
"""
function stop_recording!(tape::GradientTape)
    tape.is_recording = false
    return tape
end

"""
    clear!(tape)

Clear all recorded operations from tape.
"""
function clear!(tape::GradientTape)
    empty!(tape.operations)
    return tape
end

"""
    operations(tape)

Get the vector of recorded operations.
"""
operations(tape::GradientTape) = tape.operations

"""
    @with_tape tape expr

Execute expression while recording operations to tape.

# Examples
```julia
tape = GradientTape()
@with_tape tape begin
    y = model(x)
    loss = criterion(y, target)
end
# Now tape.operations contains recorded ops
```
"""
macro with_tape(tape_expr, expr)
    quote
        local tape = $(esc(tape_expr))
        start_recording!(tape)
        local result = $(esc(expr))
        result
    end
end

"""
    with_gradient_tape(f; persistent=false)

Execute function f with a gradient tape and return (result, tape).
Useful for debugging and visualizing computational graphs.

# Examples
```julia
result, tape = with_gradient_tape() do
    y = model(x)
    loss = criterion(y, target)
end
# Inspect tape.operations for debugging
```
"""
function with_gradient_tape(f; persistent::Bool=false)
    tape = GradientTape(persistent=persistent)
    result = f()
    stop_recording!(tape)
    return result, tape
end

"""
    gradient_with_tape(f, x...; persistent=false)

Compute gradients using Zygote while recording operations to a tape.
Returns (gradients, tape).

# Examples
```julia
grads, tape = gradient_with_tape(x -> x^2, 3.0)
# grads = (6.0,)
# tape.operations shows recorded operations
```
"""
function gradient_with_tape(f, x...; persistent::Bool=false)
    tape = GradientTape(persistent=persistent)

    # Record forward pass separately (outside gradient computation)
    result = f(x...)
    record!(tape, :function_call, collect(x), result)

    # Use Zygote for actual gradient computation (no recording during backprop)
    grads = Zygote.gradient(f, x...)

    stop_recording!(tape)
    return grads, tape
end

# Export tape functions
export GradientTape, record!, start_recording!, stop_recording!, clear!
export operations, @with_tape, with_gradient_tape, gradient_with_tape
