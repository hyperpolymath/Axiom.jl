# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Training Loop
#
# High-level training API with automatic verification.

using Zygote

"""
    train!(model, data, optimizer; kwargs...)

Train a model on data.

# Arguments
- `model`: An Axiom model
- `data`: Training data (iterable of (x, y) batches)
- `optimizer`: An optimizer instance

# Keyword Arguments
- `epochs`: Number of training epochs (default: 1)
- `loss_fn`: Loss function (default: mse_loss)
- `callbacks`: List of callback functions
- `verify`: Run verification after training (default: false)
- `verbose`: Print progress (default: true)

# Returns
Training history dictionary.
"""
function train!(
    model::Union{AbstractLayer, AxiomModel},
    data,
    optimizer::AbstractOptimizer;
    epochs::Int = 1,
    loss_fn::Function = mse_loss,
    callbacks::Vector = [],
    verify::Bool = false,
    verbose::Bool = true,
    scheduler::Union{AbstractScheduler, Nothing} = nothing
)
    history = Dict{String, Vector{Float64}}(
        "loss" => Float64[],
        "lr" => Float64[]
    )

    params = parameters(model)

    for epoch in 1:epochs
        epoch_loss = 0.0
        n_batches = 0

        for (x, y) in data
            # Compute loss and gradients
            loss, grads = Zygote.withgradient(params) do
                pred = model(Tensor(x))
                loss_fn(pred, Tensor(y))
            end

            epoch_loss += loss
            n_batches += 1

            # Update parameters
            step!(optimizer, params, grads[1])
        end

        avg_loss = epoch_loss / n_batches
        push!(history["loss"], avg_loss)
        push!(history["lr"], optimizer.lr)

        # Update learning rate
        if scheduler !== nothing
            step!(scheduler, epoch)
        end

        # Callbacks
        for callback in callbacks
            callback(epoch, avg_loss, model)
        end

        if verbose
            println("Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=4)) - LR: $(optimizer.lr)")
        end
    end

    # Post-training verification
    if verify
        @info "Running verification..."
        verify_model(model)
    end

    history
end

"""
    compute_gradients(model, x, y, loss_fn)

Compute gradients of loss with respect to model parameters.
"""
function compute_gradients(model, x, y, loss_fn)
    params = parameters(model)
    grads = Zygote.gradient(params) do
        pred = model(Tensor(x))
        loss_fn(pred, Tensor(y))
    end
    return grads[1]
end

"""
    fit!(model, train_data, val_data, optimizer; kwargs...)

Train with validation and early stopping.
"""
function fit!(
    model::Union{AbstractLayer, AxiomModel},
    train_data,
    val_data,
    optimizer::AbstractOptimizer;
    epochs::Int = 100,
    patience::Int = 10,
    loss_fn::Function = mse_loss,
    verbose::Bool = true
)
    history = Dict{String, Vector{Float64}}(
        "train_loss" => Float64[],
        "val_loss" => Float64[]
    )

    best_val_loss = Inf
    patience_counter = 0
    best_params = nothing

    for epoch in 1:epochs
        # Training
        train_loss = 0.0
        n_train = 0

        for (x, y) in train_data
            loss, grads = Zygote.withgradient(parameters(model)) do
                pred = model(Tensor(x))
                loss_fn(pred, Tensor(y))
            end
            train_loss += loss
            n_train += 1

            step!(optimizer, parameters(model), grads[1])
        end

        train_loss /= n_train

        # Validation
        val_loss = 0.0
        n_val = 0

        for (x, y) in val_data
            pred = model(Tensor(x))
            loss = loss_fn(pred, Tensor(y))
            val_loss += loss
            n_val += 1
        end

        val_loss /= n_val

        push!(history["train_loss"], train_loss)
        push!(history["val_loss"], val_loss)

        if verbose
            println("Epoch $epoch - Train Loss: $(round(train_loss, digits=4)) - Val Loss: $(round(val_loss, digits=4))")
        end

        # Early stopping
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_params = deepcopy(parameters(model))
        else
            patience_counter += 1
            if patience_counter >= patience
                @info "Early stopping at epoch $epoch"
                # Restore best parameters
                if best_params !== nothing
                    # Copy best params back to model
                    for (name, param) in pairs(parameters(model))
                        param .= best_params[name]
                    end
                end
                break
            end
        end
    end

    history
end

"""
    evaluate(model, data, metrics)

Evaluate model on data with given metrics.
"""
function evaluate(model::Union{AbstractLayer, AxiomModel}, data, metrics::Dict)
    results = Dict{String, Float64}()

    for (name, metric_fn) in metrics
        total = 0.0
        n = 0

        for (x, y) in data
            pred = model(Tensor(x))
            total += metric_fn(pred, Tensor(y))
            n += 1
        end

        results[name] = total / n
    end

    results
end

# ============================================================================
# Common Callbacks
# ============================================================================

"""
    checkpoint_callback(path; save_best_only=true)

Create a callback that saves model checkpoints.
"""
function checkpoint_callback(path::String; save_best_only::Bool=true)
    best_loss = Ref(Inf)

    function callback(epoch, loss, model)
        if save_best_only
            if loss < best_loss[]
                best_loss[] = loss
                save_model(model, joinpath(path, "best_model.axiom"))
            end
        else
            save_model(model, joinpath(path, "model_epoch_$epoch.axiom"))
        end
    end
end

"""
    early_stopping_callback(patience; min_delta=0.0)

Create an early stopping callback (alternative to fit!).
"""
function early_stopping_callback(patience::Int; min_delta::Float64=0.0)
    best_loss = Ref(Inf)
    counter = Ref(0)

    function callback(epoch, loss, model)
        if loss < best_loss[] - min_delta
            best_loss[] = loss
            counter[] = 0
        else
            counter[] += 1
            if counter[] >= patience
                throw(InterruptException())  # Stop training
            end
        end
    end
end

"""
    logging_callback(logger)

Create a callback that logs metrics.
"""
function logging_callback(log_fn::Function)
    function callback(epoch, loss, model)
        log_fn(Dict("epoch" => epoch, "loss" => loss))
    end
end

# ============================================================================
# Model Serialization
# ============================================================================

"""
    save_model(model, path)

Save model parameters to a binary file using Julia's Serialization stdlib.
The file stores a Dict mapping parameter names to their array values.

Supports round-tripping via `load_model!`.
"""
function save_model(model, path::String)
    params = parameters(model)
    param_dict = Dict{String, Any}()
    for (name, value) in pairs(params)
        param_dict[string(name)] = value
    end
    open(path, "w") do f
        Serialization.serialize(f, param_dict)
    end
    @info "Model saved to $path ($(length(param_dict)) parameters)"
end

"""
    load_model!(model, path)

Load model parameters from a binary file and restore them into the model.
The model's parameter arrays are updated in-place.
"""
function load_model!(model, path::String)
    param_dict = open(path, "r") do f
        Serialization.deserialize(f)
    end
    params = parameters(model)
    restored = 0
    for (name, value) in pairs(params)
        key = string(name)
        if haskey(param_dict, key)
            saved = param_dict[key]
            if size(saved) == size(value)
                copyto!(value, saved)
                restored += 1
            else
                @warn "Shape mismatch for $key: model=$(size(value)), file=$(size(saved))"
            end
        else
            @warn "Parameter $key not found in saved file"
        end
    end
    @info "Model loaded from $path ($restored/$(length(params)) parameters restored)"
    model
end
