# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl PyTorch Extension
#
# PyTorch model import and interoperability.
# This extension is loaded when PyCall is available.

module AxiomPyTorchExt

using Axiom
using PyCall

# Import PyTorch lazily
const torch = Ref{PyObject}()
const nn = Ref{PyObject}()

function __init__()
    try
        torch[] = pyimport("torch")
        nn[] = pyimport("torch.nn")
    catch
        @warn "PyTorch not found. PyTorch compatibility features disabled."
    end
end

"""
    from_pytorch(model_path::String) -> AxiomModel

Load a PyTorch model from file and convert to Axiom.jl.

# Arguments
- `model_path`: Path to .pt or .pth file

# Example
```julia
model = from_pytorch("resnet50.pth")
output = model(input)
```
"""
function Axiom.from_pytorch(model_path::String)
    if !isassigned(torch)
        error("PyTorch not available. Install torch: pip install torch")
    end

    # Load PyTorch model
    state_dict = torch[].load(model_path, map_location="cpu")

    # Analyze structure and convert
    _convert_state_dict(state_dict)
end

"""
    from_pytorch(model::PyObject) -> AxiomModel

Convert a live PyTorch model to Axiom.jl.
"""
function Axiom.from_pytorch(model::PyObject)
    state_dict = model.state_dict()
    _convert_state_dict(state_dict)
end

"""
Convert PyTorch state dict to Axiom.jl layers.
"""
function _convert_state_dict(state_dict)
    layers = []

    # Group parameters by layer
    layer_params = Dict{String, Dict{String, Any}}()

    for (name, param) in state_dict
        param_numpy = param.cpu().numpy()

        # Parse layer name (e.g., "layer1.conv.weight" -> "layer1.conv")
        parts = split(name, ".")
        layer_name = join(parts[1:end-1], ".")
        param_name = parts[end]

        if !haskey(layer_params, layer_name)
            layer_params[layer_name] = Dict{String, Any}()
        end
        layer_params[layer_name][param_name] = param_numpy
    end

    # Convert each layer
    for (layer_name, params) in sort(collect(layer_params), by=first)
        layer = _convert_layer(layer_name, params)
        if layer !== nothing
            push!(layers, layer)
        end
    end

    # Build sequential model
    Sequential(layers...)
end

"""
Convert a single PyTorch layer to Axiom.jl.
"""
function _convert_layer(name::String, params::Dict)
    if haskey(params, "weight")
        weight = params["weight"]

        if ndims(weight) == 2
            # Linear/Dense layer
            out_features, in_features = size(weight)
            layer = Dense(in_features, out_features)
            layer.weight .= weight'  # PyTorch uses (out, in), we use (in, out)

            if haskey(params, "bias")
                layer.bias .= params["bias"]
            end

            return layer

        elseif ndims(weight) == 4
            # Conv2D layer
            out_channels, in_channels, kH, kW = size(weight)
            layer = Conv2d(in_channels, out_channels, (kH, kW))

            # PyTorch: (out, in, kH, kW) -> Axiom: (kH, kW, in, out)
            layer.weight .= permutedims(weight, (3, 4, 2, 1))

            if haskey(params, "bias")
                layer.bias .= params["bias"]
            end

            return layer
        end

    elseif haskey(params, "running_mean")
        # BatchNorm layer
        num_features = length(params["running_mean"])
        layer = BatchNorm(num_features)

        layer.running_mean .= params["running_mean"]
        layer.running_var .= params["running_var"]

        if haskey(params, "weight")
            layer.γ .= params["weight"]
        end
        if haskey(params, "bias")
            layer.β .= params["bias"]
        end

        return layer

    elseif haskey(params, "weight") && ndims(params["weight"]) == 1 && haskey(params, "bias") && ndims(params["bias"]) == 1
        # LayerNorm (1D weight + 1D bias, no running stats)
        num_features = length(params["weight"])
        layer = LayerNorm(num_features)

        if layer.γ !== nothing
            layer.γ .= params["weight"]
        end
        if layer.β !== nothing
            layer.β .= params["bias"]
        end

        return layer
    end

    @debug "Skipping unrecognized layer" name
    nothing
end

"""
    to_pytorch(model::AxiomModel) -> PyObject

Convert an Axiom.jl model to PyTorch.
"""
function Axiom.to_pytorch(model)
    if !isassigned(torch)
        error("PyTorch not available")
    end

    # Build PyTorch module
    pytorch_layers = []

    if model isa Pipeline
        for layer in model.layers
            push!(pytorch_layers, _to_pytorch_layer(layer))
        end
    else
        push!(pytorch_layers, _to_pytorch_layer(model))
    end

    nn[].Sequential(pytorch_layers...)
end

"""
Convert an Axiom.jl layer to PyTorch.
"""
function _to_pytorch_layer(layer::Dense)
    pytorch_layer = nn[].Linear(layer.in_features, layer.out_features)
    pytorch_layer.weight.data.copy_(torch[].from_numpy(layer.weight'))
    if layer.bias !== nothing
        pytorch_layer.bias.data.copy_(torch[].from_numpy(layer.bias))
    end
    pytorch_layer
end

function _to_pytorch_layer(layer::Conv2d)
    pytorch_layer = nn[].Conv2d(
        layer.in_channels,
        layer.out_channels,
        layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding
    )
    # Convert weight format
    pytorch_layer.weight.data.copy_(
        torch[].from_numpy(permutedims(layer.weight, (4, 3, 1, 2)))
    )
    if layer.bias !== nothing
        pytorch_layer.bias.data.copy_(torch[].from_numpy(layer.bias))
    end
    pytorch_layer
end

function _to_pytorch_layer(::ReLU)
    nn[].ReLU()
end

function _to_pytorch_layer(::Sigmoid)
    nn[].Sigmoid()
end

function _to_pytorch_layer(::Softmax)
    nn[].Softmax(dim=-1)
end

function _to_pytorch_layer(layer::BatchNorm)
    pytorch_layer = nn[].BatchNorm1d(layer.num_features)
    pytorch_layer.weight.data.copy_(torch[].from_numpy(layer.γ))
    pytorch_layer.bias.data.copy_(torch[].from_numpy(layer.β))
    pytorch_layer.running_mean.copy_(torch[].from_numpy(layer.running_mean))
    pytorch_layer.running_var.copy_(torch[].from_numpy(layer.running_var))
    pytorch_layer
end

function _to_pytorch_layer(::Flatten)
    nn[].Flatten()
end

function _to_pytorch_layer(layer::LayerNorm)
    shape = layer.normalized_shape isa Int ? (layer.normalized_shape,) : layer.normalized_shape
    pytorch_layer = nn[].LayerNorm(shape)
    if layer.γ !== nothing
        pytorch_layer.weight.data.copy_(torch[].from_numpy(vec(layer.γ)))
    end
    if layer.β !== nothing
        pytorch_layer.bias.data.copy_(torch[].from_numpy(vec(layer.β)))
    end
    pytorch_layer
end

function _to_pytorch_layer(layer)
    @warn "Unknown layer type $(typeof(layer)) for PyTorch export, wrapping as Identity"
    nn[].Identity()
end

"""
    verify_pytorch_compatibility(axiom_model, pytorch_model, input)

Verify that Axiom.jl and PyTorch models produce the same output.
"""
function verify_pytorch_compatibility(axiom_model, pytorch_model, input::AbstractArray)
    # Run Axiom.jl model
    axiom_output = axiom_model(input)

    # Run PyTorch model
    input_tensor = torch[].from_numpy(input).float()
    pytorch_model.eval()
    pytorch_output = pytorch_model(input_tensor).detach().numpy()

    # Compare
    max_diff = maximum(abs.(axiom_output .- pytorch_output))
    is_close = max_diff < 1e-5

    Dict(
        "compatible" => is_close,
        "max_difference" => max_diff,
        "axiom_output" => axiom_output,
        "pytorch_output" => pytorch_output
    )
end

# Export functions
export from_pytorch, to_pytorch, verify_pytorch_compatibility

end # module
