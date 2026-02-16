# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl interop APIs:
# - from_pytorch(...) import API
# - to_onnx(...) export API

const _PYTORCH_EXPORT_FORMAT = "axiom.pytorch.sequential.v1"

function _string_key_dict(input::AbstractDict)
    out = Dict{String, Any}()
    for (k, v) in input
        out[String(k)] = v
    end
    out
end

function _float32_vector(value, field_name::AbstractString)
    value isa AbstractVector || throw(ArgumentError("`$field_name` must be a vector"))
    out = Vector{Float32}(undef, length(value))
    for i in eachindex(value)
        v = value[i]
        v isa Number || throw(ArgumentError("`$field_name[$i]` must be numeric"))
        out[i] = Float32(v)
    end
    out
end

function _float32_rows(value, field_name::AbstractString)
    value isa AbstractVector || throw(ArgumentError("`$field_name` must be a vector of vectors"))
    rows = Vector{Vector{Float32}}(undef, length(value))
    for i in eachindex(value)
        row = value[i]
        row isa AbstractVector || throw(ArgumentError("`$field_name[$i]` must be a vector"))
        rows[i] = _float32_vector(row, "$field_name[$i]")
    end
    rows
end

function _tuple2(value, field_name::AbstractString)
    if value isa Integer
        return (Int(value), Int(value))
    end
    value isa AbstractVector || throw(ArgumentError("`$field_name` must be an integer or pair"))
    length(value) == 2 || throw(ArgumentError("`$field_name` must have length 2"))
    (Int(value[1]), Int(value[2]))
end

function _nested_shape(value, field_name::AbstractString)
    if value isa Number
        return ()
    end
    value isa AbstractVector || throw(ArgumentError("`$field_name` must be numeric or nested numeric vectors"))
    isempty(value) && throw(ArgumentError("`$field_name` must not be empty"))
    subshape = _nested_shape(value[1], "$field_name[1]")
    for i in 2:length(value)
        _nested_shape(value[i], "$field_name[$i]") == subshape ||
            throw(ArgumentError("`$field_name` has ragged nested dimensions"))
    end
    (length(value), subshape...)
end

function _flatten_nested_numbers!(out::Vector{Float32}, value, field_name::AbstractString)
    if value isa Number
        push!(out, Float32(value))
        return
    end
    value isa AbstractVector || throw(ArgumentError("`$field_name` must be numeric or nested numeric vectors"))
    for (i, v) in enumerate(value)
        _flatten_nested_numbers!(out, v, "$field_name[$i]")
    end
end

function _array_from_row_major(flat::Vector{Float32}, dims::Tuple)
    arr = Array{Float32}(undef, dims...)
    next_idx = Ref(1)
    function fill_row_major(prefix::Tuple, depth::Int)
        if depth > length(dims)
            arr[prefix...] = flat[next_idx[]]
            next_idx[] += 1
            return
        end
        for i in 1:dims[depth]
            fill_row_major((prefix..., i), depth + 1)
        end
    end
    fill_row_major((), 1)
    arr
end

function _nested_array_f32(value, field_name::AbstractString)
    shape = _nested_shape(value, field_name)
    flat = Float32[]
    _flatten_nested_numbers!(flat, value, field_name)
    _array_from_row_major(flat, shape)
end

function _default_pytorch_bridge_script()
    normpath(joinpath(@__DIR__, "..", "..", "scripts", "pytorch_to_axiom_descriptor.py"))
end

function _run_pytorch_bridge(
    input_path::AbstractString;
    python_cmd::AbstractString = get(ENV, "AXIOM_PYTHON", "python3"),
    bridge_script::AbstractString = _default_pytorch_bridge_script(),
    strict::Bool = true,
)
    isfile(input_path) || throw(ArgumentError("Checkpoint path does not exist: $(input_path)"))
    isfile(bridge_script) || throw(ArgumentError("PyTorch bridge script not found: $(bridge_script)"))

    parts = Base.shell_split(String(python_cmd))
    isempty(parts) && throw(ArgumentError("`python_cmd` must not be empty"))

    tmp_path, tmp_io = mktemp()
    close(tmp_io)
    tmp_json_path = tmp_path * ".pytorch.json"
    mv(tmp_path, tmp_json_path; force=true)

    cmd_parts = copy(parts)
    append!(cmd_parts, [
        String(bridge_script),
        "--input", String(input_path),
        "--output", String(tmp_json_path),
        strict ? "--strict" : "--no-strict",
    ])
    cmd = Cmd(cmd_parts)

    stdout_buf = IOBuffer()
    stderr_buf = IOBuffer()
    ok = success(pipeline(cmd; stdout=stdout_buf, stderr=stderr_buf))
    if !ok
        stderr_out = String(take!(stderr_buf))
        stdout_out = String(take!(stdout_buf))
        rm(tmp_json_path; force=true)
        throw(ErrorException(
            "PyTorch bridge failed for `$(input_path)`.\n" *
            "Command: $(cmd)\n" *
            (isempty(stdout_out) ? "" : "stdout:\n$(stdout_out)\n") *
            (isempty(stderr_out) ? "" : "stderr:\n$(stderr_out)\n")
        ))
    end

    isfile(tmp_json_path) || throw(ErrorException("PyTorch bridge did not produce descriptor output: $(tmp_json_path)"))
    tmp_json_path
end

function _pytorch_layers(spec::Dict{String, Any})
    raw_layers = get(spec, "layers", get(spec, "modules", nothing))
    raw_layers isa AbstractVector || throw(ArgumentError("PyTorch spec must contain `layers` (or `modules`) as a vector"))
    collect(raw_layers)
end

function _linear_from_pytorch(spec::Dict{String, Any})
    in_features = Int(get(spec, "in_features", 0))
    out_features = Int(get(spec, "out_features", 0))
    in_features > 0 || throw(ArgumentError("Linear layer requires positive `in_features`"))
    out_features > 0 || throw(ArgumentError("Linear layer requires positive `out_features`"))

    weight_rows = _float32_rows(get(spec, "weight", nothing), "weight")
    length(weight_rows) == out_features || throw(ArgumentError("Linear `weight` must have $out_features rows (PyTorch shape: [out_features, in_features])"))
    all(length(row) == in_features for row in weight_rows) || throw(ArgumentError("Each `weight` row must have $in_features elements"))

    has_bias = haskey(spec, "bias") && spec["bias"] !== nothing
    bias_values = has_bias ? _float32_vector(spec["bias"], "bias") : Float32[]
    if has_bias
        length(bias_values) == out_features || throw(ArgumentError("Linear `bias` must have $out_features elements"))
    end

    # PyTorch Linear stores weights as [out_features, in_features].
    # Axiom Dense expects [in_features, out_features].
    dense = Dense(in_features, out_features, identity; bias=has_bias, dtype=Float32)
    for out_idx in 1:out_features
        for in_idx in 1:in_features
            dense.weight[in_idx, out_idx] = weight_rows[out_idx][in_idx]
        end
    end
    if has_bias
        dense.bias .= bias_values
    end
    dense
end

function _conv2d_from_pytorch(spec::Dict{String, Any})
    in_channels = Int(get(spec, "in_channels", 0))
    out_channels = Int(get(spec, "out_channels", 0))
    in_channels > 0 || throw(ArgumentError("Conv2d layer requires positive `in_channels`"))
    out_channels > 0 || throw(ArgumentError("Conv2d layer requires positive `out_channels`"))

    kernel = _tuple2(get(spec, "kernel_size", nothing), "kernel_size")
    stride = _tuple2(get(spec, "stride", [1, 1]), "stride")
    padding = _tuple2(get(spec, "padding", [0, 0]), "padding")
    dilation = _tuple2(get(spec, "dilation", [1, 1]), "dilation")
    groups = Int(get(spec, "groups", 1))
    groups > 0 || throw(ArgumentError("Conv2d layer requires positive `groups`"))

    has_bias = haskey(spec, "bias") && spec["bias"] !== nothing
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel;
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=has_bias,
        dtype=Float32
    )

    weight_oihw = _nested_array_f32(get(spec, "weight", nothing), "weight")
    expected = (out_channels, div(in_channels, groups), kernel[1], kernel[2])
    size(weight_oihw) == expected || throw(ArgumentError(
        "Conv2d `weight` must have shape $(expected) in PyTorch (O, I/groups, kH, kW), got $(size(weight_oihw))"
    ))
    conv.weight .= permutedims(weight_oihw, (3, 4, 2, 1)) # (kH, kW, I/groups, O)

    if has_bias
        bias_values = _float32_vector(spec["bias"], "bias")
        length(bias_values) == out_channels || throw(ArgumentError("Conv2d `bias` must have $out_channels elements"))
        conv.bias .= bias_values
    end
    conv
end

function _batchnorm_from_pytorch(spec::Dict{String, Any})
    num_features = Int(get(spec, "num_features", 0))
    num_features > 0 || throw(ArgumentError("BatchNorm layer requires positive `num_features`"))
    affine = Bool(get(spec, "affine", true))
    track_running_stats = Bool(get(spec, "track_running_stats", true))
    momentum = Float32(get(spec, "momentum", 0.1))
    eps = Float32(get(spec, "eps", 1e-5))

    bn = BatchNorm(
        num_features;
        momentum=momentum,
        eps=eps,
        affine=affine,
        track_running_stats=track_running_stats,
        dtype=Float32
    )

    if affine
        gamma_values = _float32_vector(get(spec, "weight", nothing), "weight")
        beta_values = _float32_vector(get(spec, "bias", nothing), "bias")
        length(gamma_values) == num_features || throw(ArgumentError("BatchNorm `weight` must have $num_features elements"))
        length(beta_values) == num_features || throw(ArgumentError("BatchNorm `bias` must have $num_features elements"))
        bn.γ .= gamma_values
        bn.β .= beta_values
    end

    if haskey(spec, "running_mean") && spec["running_mean"] !== nothing
        running_mean = _float32_vector(spec["running_mean"], "running_mean")
        length(running_mean) == num_features || throw(ArgumentError("BatchNorm `running_mean` must have $num_features elements"))
        bn.running_mean .= running_mean
    end
    if haskey(spec, "running_var") && spec["running_var"] !== nothing
        running_var = _float32_vector(spec["running_var"], "running_var")
        length(running_var) == num_features || throw(ArgumentError("BatchNorm `running_var` must have $num_features elements"))
        bn.running_var .= running_var
    end
    bn
end

function _layernorm_from_pytorch(spec::Dict{String, Any})
    raw_shape = get(spec, "normalized_shape", nothing)
    norm_shape = if raw_shape isa Integer
        (Int(raw_shape),)
    elseif raw_shape isa AbstractVector
        Tuple(Int(v) for v in raw_shape)
    else
        throw(ArgumentError("LayerNorm requires `normalized_shape`"))
    end
    elementwise_affine = Bool(get(spec, "elementwise_affine", true))
    eps = Float32(get(spec, "eps", 1e-5))

    ln = LayerNorm(norm_shape; eps=eps, elementwise_affine=elementwise_affine, dtype=Float32)
    if elementwise_affine
        gamma = _nested_array_f32(get(spec, "weight", nothing), "weight")
        beta = _nested_array_f32(get(spec, "bias", nothing), "bias")
        size(gamma) == norm_shape || throw(ArgumentError("LayerNorm `weight` shape mismatch: expected $(norm_shape), got $(size(gamma))"))
        size(beta) == norm_shape || throw(ArgumentError("LayerNorm `bias` shape mismatch: expected $(norm_shape), got $(size(beta))"))
        ln.γ .= gamma
        ln.β .= beta
    end
    ln
end

function _pytorch_softmax_dim(spec::Dict{String, Any})
    dim = Int(get(spec, "dim", -1))
    dim < 0 ? -1 : dim + 1
end

function _pytorch_module_to_layer(spec::Dict{String, Any}; strict::Bool = true)
    module_type = lowercase(String(get(spec, "type", get(spec, "module", ""))))
    if isempty(module_type)
        strict && throw(ArgumentError("PyTorch module entries require a `type` field"))
        return nothing
    end

    if module_type in ("linear", "torch.nn.linear")
        return _linear_from_pytorch(spec)
    elseif module_type in ("conv2d", "torch.nn.conv2d")
        return _conv2d_from_pytorch(spec)
    elseif module_type in ("batchnorm", "batchnorm2d", "torch.nn.batchnorm2d")
        return _batchnorm_from_pytorch(spec)
    elseif module_type in ("layernorm", "torch.nn.layernorm")
        return _layernorm_from_pytorch(spec)
    elseif module_type in ("maxpool2d", "torch.nn.maxpool2d")
        return MaxPool2d(
            _tuple2(get(spec, "kernel_size", nothing), "kernel_size");
            stride=_tuple2(get(spec, "stride", get(spec, "kernel_size", nothing)), "stride"),
            padding=_tuple2(get(spec, "padding", [0, 0]), "padding")
        )
    elseif module_type in ("avgpool2d", "torch.nn.avgpool2d")
        return AvgPool2d(
            _tuple2(get(spec, "kernel_size", nothing), "kernel_size");
            stride=_tuple2(get(spec, "stride", get(spec, "kernel_size", nothing)), "stride"),
            padding=_tuple2(get(spec, "padding", [0, 0]), "padding"),
            count_include_pad=Bool(get(spec, "count_include_pad", true))
        )
    elseif module_type in ("adaptiveavgpool2d", "torch.nn.adaptiveavgpool2d")
        return AdaptiveAvgPool2d(_tuple2(get(spec, "output_size", nothing), "output_size"))
    elseif module_type in ("globalavgpool", "globalaveragepool")
        return GlobalAvgPool()
    elseif module_type in ("relu", "torch.nn.relu")
        return ReLU()
    elseif module_type in ("sigmoid", "torch.nn.sigmoid")
        return Sigmoid()
    elseif module_type in ("tanh", "torch.nn.tanh")
        return Tanh()
    elseif module_type in ("softmax", "torch.nn.softmax")
        return Softmax(dims=_pytorch_softmax_dim(spec))
    elseif module_type in ("leakyrelu", "torch.nn.leakyrelu")
        alpha = Float32(get(spec, "negative_slope", 0.01))
        return LeakyReLU(alpha)
    elseif module_type in ("flatten", "torch.nn.flatten")
        start_dim = Int(get(spec, "start_dim", 1))
        end_dim = Int(get(spec, "end_dim", -1))
        if strict && end_dim != -1
            throw(ArgumentError("Flatten import currently supports only end_dim=-1 (got end_dim=$end_dim)"))
        end
        return Flatten(start_dim=start_dim + 1) # PyTorch uses zero-based dim
    elseif module_type in ("identity", "torch.nn.identity", "dropout", "torch.nn.dropout")
        return nothing
    end

    strict && throw(ArgumentError("Unsupported PyTorch module type: `$module_type`"))
    @warn "Skipping unsupported PyTorch module type `$module_type`"
    nothing
end

"""
    from_pytorch(path::AbstractString; strict=true, bridge=true, python_cmd="python3", bridge_script=scripts/pytorch_to_axiom_descriptor.py)

Import a model from:
- a PyTorch JSON descriptor (`axiom.pytorch.sequential.v1`)
- or a `.pt`/`.pth`/`.ckpt` checkpoint via the built-in Python bridge script.

Bridge requirements for direct checkpoints:
- Python runtime (`python3` by default, configurable via `python_cmd`)
- `torch` installed in that Python environment

Supported descriptor format:
- `format = "axiom.pytorch.sequential.v1"`
- `layers = [...]` where each layer is a module object (`Linear`, `ReLU`, etc.)
"""
function from_pytorch(
    path::AbstractString;
    strict::Bool = true,
    bridge::Bool = true,
    python_cmd::AbstractString = get(ENV, "AXIOM_PYTHON", "python3"),
    bridge_script::AbstractString = _default_pytorch_bridge_script(),
)
    isfile(path) || throw(ArgumentError("PyTorch import path does not exist: $(path)"))
    ext = lowercase(splitext(String(path))[2])

    if ext in (".pt", ".pth", ".ckpt")
        bridge || throw(ArgumentError(
            "Direct `.pt/.pth/.ckpt` loading requires bridge support. " *
            "Pass `bridge=true` (default) or export JSON descriptor first."
        ))
        descriptor_path = _run_pytorch_bridge(
            path;
            python_cmd=python_cmd,
            bridge_script=bridge_script,
            strict=strict
        )
        try
            return from_pytorch(descriptor_path; strict=strict, bridge=false)
        finally
            rm(descriptor_path; force=true)
        end
    end

    spec = try
        JSON.parse(read(path, String))
    catch e
        rethrow(e)
    end

    from_pytorch(spec; strict=strict)
end

"""
    from_pytorch(spec::AbstractDict; strict=true)

Import a model from an in-memory PyTorch JSON descriptor.
"""
function from_pytorch(spec::AbstractDict; strict::Bool = true)
    parsed = _string_key_dict(spec)
    format_name = String(get(parsed, "format", _PYTORCH_EXPORT_FORMAT))
    if format_name != _PYTORCH_EXPORT_FORMAT
        strict && throw(ArgumentError("Unsupported PyTorch descriptor format `$format_name` (expected `$(_PYTORCH_EXPORT_FORMAT)`)"))
        @warn "Importing non-canonical PyTorch descriptor format `$format_name`"
    end

    layers = AbstractLayer[]
    for (idx, raw_layer) in enumerate(_pytorch_layers(parsed))
        raw_layer isa AbstractDict || throw(ArgumentError("`layers[$idx]` must be a module object"))
        layer = _pytorch_module_to_layer(_string_key_dict(raw_layer); strict=strict)
        layer === nothing && continue
        push!(layers, layer)
    end

    isempty(layers) && throw(ArgumentError("No importable layers found in PyTorch descriptor"))
    Sequential(layers...)
end

load_pytorch(path::AbstractString; kwargs...) = from_pytorch(path; kwargs...)
load_pytorch(spec::AbstractDict; kwargs...) = from_pytorch(spec; kwargs...)

const _PB_WIRE_VARINT = 0
const _PB_WIRE_LEN = 2
const _PB_WIRE_32 = 5

const _ONNX_TENSOR_FLOAT = 1
const _ONNX_ATTR_FLOAT = 1
const _ONNX_ATTR_INT = 2
const _ONNX_ATTR_INTS = 7
const _ONNX_IR_VERSION = 8

function _pb_varint(value::Integer)
    u = value >= 0 ? UInt64(value) : reinterpret(UInt64, Int64(value))
    out = UInt8[]
    while u >= 0x80
        push!(out, UInt8((u & 0x7f) | 0x80))
        u >>= 7
    end
    push!(out, UInt8(u))
    out
end

_pb_key(field::Integer, wire::Integer) = _pb_varint((UInt64(field) << 3) | UInt64(wire))

function _pb_field_varint(field::Integer, value::Integer)
    out = UInt8[]
    append!(out, _pb_key(field, _PB_WIRE_VARINT))
    append!(out, _pb_varint(value))
    out
end

function _pb_field_len(field::Integer, payload::Vector{UInt8})
    out = UInt8[]
    append!(out, _pb_key(field, _PB_WIRE_LEN))
    append!(out, _pb_varint(length(payload)))
    append!(out, payload)
    out
end

function _pb_field_string(field::Integer, value::AbstractString)
    _pb_field_len(field, Vector{UInt8}(codeunits(String(value))))
end

function _pb_field_bytes(field::Integer, value::Vector{UInt8})
    _pb_field_len(field, value)
end

function _pb_field_float32(field::Integer, value::Real)
    out = UInt8[]
    append!(out, _pb_key(field, _PB_WIRE_32))
    append!(out, collect(reinterpret(UInt8, [Float32(value)])))
    out
end

function _onnx_attr_int(name::AbstractString, value::Integer)
    msg = UInt8[]
    append!(msg, _pb_field_string(1, name))
    append!(msg, _pb_field_varint(3, value))
    append!(msg, _pb_field_varint(20, _ONNX_ATTR_INT))
    msg
end

function _onnx_attr_float(name::AbstractString, value::Real)
    msg = UInt8[]
    append!(msg, _pb_field_string(1, name))
    append!(msg, _pb_field_float32(2, value))
    append!(msg, _pb_field_varint(20, _ONNX_ATTR_FLOAT))
    msg
end

function _onnx_attr_ints(name::AbstractString, values::Vector{Int})
    msg = UInt8[]
    append!(msg, _pb_field_string(1, name))
    for value in values
        append!(msg, _pb_field_varint(8, value))
    end
    append!(msg, _pb_field_varint(20, _ONNX_ATTR_INTS))
    msg
end

function _row_major_f32_nd(arr::AbstractArray)
    out = Vector{Float32}(undef, length(arr))
    idx = Ref(1)
    dims = size(arr)
    function emit(prefix::Tuple, depth::Int)
        if depth > length(dims)
            out[idx[]] = Float32(arr[prefix...])
            idx[] += 1
            return
        end
        for i in 1:dims[depth]
            emit((prefix..., i), depth + 1)
        end
    end
    emit((), 1)
    out
end

_row_major_f32(mat::AbstractMatrix) = _row_major_f32_nd(mat)

function _onnx_tensor_f32(name::AbstractString, dims::Vector{Int}, data::Vector{Float32})
    msg = UInt8[]
    for dim in dims
        append!(msg, _pb_field_varint(1, dim))
    end
    append!(msg, _pb_field_varint(2, _ONNX_TENSOR_FLOAT))
    append!(msg, _pb_field_string(7, name))
    append!(msg, _pb_field_bytes(8, collect(reinterpret(UInt8, data))))
    msg
end

function _onnx_dim(dim)
    msg = UInt8[]
    if dim isa Integer
        append!(msg, _pb_field_varint(1, Int(dim)))
    else
        dim_name = dim === :dynamic ? "dynamic" : String(dim)
        append!(msg, _pb_field_string(2, dim_name))
    end
    msg
end

function _onnx_value_info(name::AbstractString, dims::Vector)
    shape = UInt8[]
    for dim in dims
        append!(shape, _pb_field_len(1, _onnx_dim(dim)))
    end

    tensor_type = UInt8[]
    append!(tensor_type, _pb_field_varint(1, _ONNX_TENSOR_FLOAT))
    append!(tensor_type, _pb_field_len(2, shape))

    type_proto = UInt8[]
    append!(type_proto, _pb_field_len(1, tensor_type))

    value_info = UInt8[]
    append!(value_info, _pb_field_string(1, name))
    append!(value_info, _pb_field_len(2, type_proto))
    value_info
end

function _onnx_node(
    op_type::AbstractString;
    inputs::Vector{String} = String[],
    outputs::Vector{String} = String[],
    name::AbstractString = "",
    attrs::Vector{Vector{UInt8}} = Vector{Vector{UInt8}}()
)
    msg = UInt8[]
    for input_name in inputs
        append!(msg, _pb_field_string(1, input_name))
    end
    for output_name in outputs
        append!(msg, _pb_field_string(2, output_name))
    end
    isempty(name) || append!(msg, _pb_field_string(3, name))
    append!(msg, _pb_field_string(4, op_type))
    for attr in attrs
        append!(msg, _pb_field_len(5, attr))
    end
    msg
end

function _onnx_operator_set(version::Integer)
    msg = UInt8[]
    append!(msg, _pb_field_varint(2, version))
    msg
end

function _onnx_graph(
    name::AbstractString,
    nodes::Vector{Vector{UInt8}},
    initializers::Vector{Vector{UInt8}},
    inputs::Vector{Vector{UInt8}},
    outputs::Vector{Vector{UInt8}}
)
    msg = UInt8[]
    for node in nodes
        append!(msg, _pb_field_len(1, node))
    end
    append!(msg, _pb_field_string(2, name))
    for initializer in initializers
        append!(msg, _pb_field_len(5, initializer))
    end
    for input in inputs
        append!(msg, _pb_field_len(11, input))
    end
    for output in outputs
        append!(msg, _pb_field_len(12, output))
    end
    msg
end

function _onnx_model(
    graph::Vector{UInt8};
    opset::Integer = 17,
    producer_name::AbstractString = "Axiom.jl",
    producer_version::AbstractString = string(VERSION),
    model_version::Integer = 1
)
    msg = UInt8[]
    append!(msg, _pb_field_varint(1, _ONNX_IR_VERSION))
    append!(msg, _pb_field_len(2, _onnx_operator_set(opset)))
    append!(msg, _pb_field_string(3, producer_name))
    append!(msg, _pb_field_string(4, producer_version))
    append!(msg, _pb_field_string(5, "axiom"))
    append!(msg, _pb_field_varint(6, model_version))
    append!(msg, _pb_field_len(8, graph))
    msg
end

function _export_layers(model)
    if model isa Pipeline
        return collect(model.layers)
    elseif model isa AbstractLayer
        return AbstractLayer[model]
    end
    throw(ArgumentError("`to_onnx` expects an `AbstractLayer` model (for example `Sequential(...)`)"))
end

function _normalize_dim_value(dim)
    dim isa Integer ? Int(dim) : dim
end

function _shape_product(dims)
    if isempty(dims)
        return 1
    end
    all(d -> d isa Integer, dims) || return :dynamic
    prod(Int(d) for d in dims)
end

function _conv_pool_out_dim(dim, pad::Int, kernel::Int, stride::Int)
    dim isa Integer || return :dynamic
    div(Int(dim) + 2 * pad - kernel, stride) + 1
end

function _normalize_input_shape(input_shape, layers::Vector{AbstractLayer})
    if input_shape === nothing
        first = layers[1]
        if first isa Dense
            return Any[:batch, first.in_features]
        end
        throw(ArgumentError(
            "`input_shape` is required for non-Dense-first models. " *
            "Example for image models: `input_shape=(1, 224, 224, C)`"
        ))
    elseif input_shape isa Integer
        return Any[:batch, Int(input_shape)]
    end

    shape_tuple = input_shape isa Tuple ? input_shape : Tuple(input_shape)
    isempty(shape_tuple) && throw(ArgumentError("`input_shape` must not be empty"))

    normalized = Any[_normalize_dim_value(d) for d in shape_tuple]
    if length(normalized) == 1
        return Any[:batch, normalized[1]]
    elseif length(normalized) == 3
        # Allow HWC shorthand for image inputs.
        return Any[:batch, normalized...]
    end
    normalized
end

function _dense_activation_spec(activation)
    if activation === identity
        return nothing
    elseif activation === relu
        return ("Relu", Vector{Vector{UInt8}}())
    elseif activation === sigmoid
        return ("Sigmoid", Vector{Vector{UInt8}}())
    elseif activation === tanh
        return ("Tanh", Vector{Vector{UInt8}}())
    elseif activation === softmax
        return ("Softmax", [_onnx_attr_int("axis", -1)])
    end
    throw(ArgumentError("Dense activation `$(activation)` is not ONNX-exportable in the current subset"))
end

function _layer_activation_spec(layer::AbstractLayer)
    if layer isa ReLU
        return ("Relu", Vector{Vector{UInt8}}())
    elseif layer isa Sigmoid
        return ("Sigmoid", Vector{Vector{UInt8}}())
    elseif layer isa Tanh
        return ("Tanh", Vector{Vector{UInt8}}())
    elseif layer isa Softmax
        axis = layer.dims < 0 ? -1 : layer.dims - 1
        return ("Softmax", [_onnx_attr_int("axis", axis)])
    elseif layer isa LeakyReLU
        return ("LeakyRelu", [_onnx_attr_float("alpha", layer.α)])
    elseif layer isa Flatten
        axis = max(layer.start_dim - 1, 0)
        return ("Flatten", [_onnx_attr_int("axis", axis)])
    end
    throw(ArgumentError("Layer `$(typeof(layer))` is not ONNX-exportable in the current subset"))
end

function _to_onnx_bytes(
    model;
    input_shape = nothing,
    opset::Integer = 17,
    producer_name::AbstractString = "Axiom.jl",
    model_name::AbstractString = "AxiomModel"
)
    opset > 0 || throw(ArgumentError("`opset` must be positive"))

    layers = _export_layers(model)
    isempty(layers) && throw(ArgumentError("Cannot export an empty model"))

    input_dims = _normalize_input_shape(input_shape, layers)
    current_dims = copy(input_dims)

    nodes = Vector{Vector{UInt8}}()
    initializers = Vector{Vector{UInt8}}()
    current_name = "input"
    dense_index = 0
    node_index = 0

    function next_name(prefix::AbstractString)
        node_index += 1
        "$(prefix)_$(node_index)"
    end

    function add_node!(
        op_type::AbstractString,
        inputs::Vector{String},
        outputs::Vector{String};
        attrs::Vector{Vector{UInt8}} = Vector{Vector{UInt8}}(),
        name_prefix::AbstractString = lowercase(op_type),
    )
        push!(nodes, _onnx_node(
            op_type;
            inputs=inputs,
            outputs=outputs,
            name=next_name(name_prefix),
            attrs=attrs
        ))
    end

    function nhwc_to_nchw(input_name::String)
        out = next_name("nhwc_to_nchw")
        push!(nodes, _onnx_node(
            "Transpose";
            inputs=[input_name],
            outputs=[out],
            name=next_name("transpose"),
            attrs=[_onnx_attr_ints("perm", [0, 3, 1, 2])]
        ))
        out
    end

    function nchw_to_nhwc(input_name::String)
        out = next_name("nchw_to_nhwc")
        push!(nodes, _onnx_node(
            "Transpose";
            inputs=[input_name],
            outputs=[out],
            name=next_name("transpose"),
            attrs=[_onnx_attr_ints("perm", [0, 2, 3, 1])]
        ))
        out
    end

    for layer in layers
        if layer isa Dense
            length(current_dims) == 2 || throw(ArgumentError(
                "Dense export expects rank-2 input, got rank $(length(current_dims)). " *
                "Add `Flatten()` before Dense."
            ))
            if current_dims[end] isa Integer && Int(current_dims[end]) != layer.in_features
                throw(DimensionMismatch(
                    "Dense layer expects $(layer.in_features) features, got $(current_dims[end]) from previous layer"
                ))
            end

            dense_index += 1
            weight_name = "dense_$(dense_index)_weight"
            bias_name = "dense_$(dense_index)_bias"
            dense_out = "dense_$(dense_index)_out"

            push!(initializers, _onnx_tensor_f32(
                weight_name,
                [size(layer.weight, 1), size(layer.weight, 2)],
                _row_major_f32(layer.weight)
            ))

            gemm_inputs = String[current_name, weight_name]
            if layer.bias !== nothing
                push!(initializers, _onnx_tensor_f32(
                    bias_name,
                    [length(layer.bias)],
                    Float32.(layer.bias)
                ))
                push!(gemm_inputs, bias_name)
            end

            add_node!("Gemm", gemm_inputs, [dense_out]; name_prefix="gemm")
            current_name = dense_out
            current_dims = Any[current_dims[1], layer.out_features]

            dense_activation = _dense_activation_spec(layer.activation)
            if dense_activation !== nothing
                op_type, attrs = dense_activation
                act_out = next_name("act_out")
                add_node!(op_type, [current_name], [act_out]; attrs=attrs, name_prefix="act")
                current_name = act_out
            end
        elseif layer isa Conv2d
            length(current_dims) == 4 || throw(ArgumentError("Conv2d export expects NHWC rank-4 input"))
            if current_dims[4] isa Integer && Int(current_dims[4]) != layer.in_channels
                throw(DimensionMismatch(
                    "Conv2d expects $(layer.in_channels) channels, got $(current_dims[4])"
                ))
            end

            conv_in = nhwc_to_nchw(current_name)
            weight_name = next_name("conv_weight")
            bias_name = next_name("conv_bias")
            conv_out_nchw = next_name("conv_out")

            conv_weight_oihw = permutedims(layer.weight, (4, 3, 1, 2))
            push!(initializers, _onnx_tensor_f32(
                weight_name,
                Int[size(conv_weight_oihw)...],
                _row_major_f32_nd(conv_weight_oihw)
            ))

            conv_inputs = String[conv_in, weight_name]
            if layer.bias !== nothing
                push!(initializers, _onnx_tensor_f32(
                    bias_name,
                    [length(layer.bias)],
                    Float32.(layer.bias)
                ))
                push!(conv_inputs, bias_name)
            end

            add_node!(
                "Conv",
                conv_inputs,
                [conv_out_nchw];
                attrs=[
                    _onnx_attr_ints("kernel_shape", [layer.kernel_size[1], layer.kernel_size[2]]),
                    _onnx_attr_ints("strides", [layer.stride[1], layer.stride[2]]),
                    _onnx_attr_ints("pads", [layer.padding[1], layer.padding[2], layer.padding[1], layer.padding[2]]),
                    _onnx_attr_ints("dilations", [layer.dilation[1], layer.dilation[2]]),
                    _onnx_attr_int("group", layer.groups),
                ],
                name_prefix="conv"
            )

            current_name = nchw_to_nhwc(conv_out_nchw)
            current_dims = Any[
                current_dims[1],
                _conv_pool_out_dim(current_dims[2], layer.padding[1], layer.kernel_size[1], layer.stride[1]),
                _conv_pool_out_dim(current_dims[3], layer.padding[2], layer.kernel_size[2], layer.stride[2]),
                layer.out_channels,
            ]
        elseif layer isa BatchNorm
            rank = length(current_dims)
            rank in (2, 4) || throw(ArgumentError("BatchNorm export supports rank-2 or rank-4 tensors"))
            nfeatures = rank == 2 ? current_dims[2] : current_dims[4]
            if nfeatures isa Integer && Int(nfeatures) != layer.num_features
                throw(DimensionMismatch("BatchNorm expects $(layer.num_features) features/channels, got $nfeatures"))
            end

            scale_name = next_name("bn_scale")
            bias_name = next_name("bn_bias")
            mean_name = next_name("bn_mean")
            var_name = next_name("bn_var")

            gamma = layer.affine ? Float32.(layer.γ) : ones(Float32, layer.num_features)
            beta = layer.affine ? Float32.(layer.β) : zeros(Float32, layer.num_features)
            push!(initializers, _onnx_tensor_f32(scale_name, [length(gamma)], gamma))
            push!(initializers, _onnx_tensor_f32(bias_name, [length(beta)], beta))
            push!(initializers, _onnx_tensor_f32(mean_name, [length(layer.running_mean)], Float32.(layer.running_mean)))
            push!(initializers, _onnx_tensor_f32(var_name, [length(layer.running_var)], Float32.(layer.running_var)))

            if rank == 4
                bn_in = nhwc_to_nchw(current_name)
                bn_out_nchw = next_name("bn_out")
                add_node!(
                    "BatchNormalization",
                    [bn_in, scale_name, bias_name, mean_name, var_name],
                    [bn_out_nchw];
                    attrs=[
                        _onnx_attr_float("epsilon", layer.eps),
                        _onnx_attr_float("momentum", layer.momentum),
                    ],
                    name_prefix="batchnorm"
                )
                current_name = nchw_to_nhwc(bn_out_nchw)
            else
                bn_out = next_name("bn_out")
                add_node!(
                    "BatchNormalization",
                    [current_name, scale_name, bias_name, mean_name, var_name],
                    [bn_out];
                    attrs=[
                        _onnx_attr_float("epsilon", layer.eps),
                        _onnx_attr_float("momentum", layer.momentum),
                    ],
                    name_prefix="batchnorm"
                )
                current_name = bn_out
            end
        elseif layer isa LayerNorm
            norm_ndims = length(layer.normalized_shape)
            length(current_dims) >= norm_ndims || throw(ArgumentError("LayerNorm normalized_shape is incompatible with current rank"))

            scale = layer.elementwise_affine ? Float32.(layer.γ) : ones(Float32, layer.normalized_shape...)
            bias = layer.elementwise_affine ? Float32.(layer.β) : zeros(Float32, layer.normalized_shape...)
            scale_name = next_name("ln_scale")
            bias_name = next_name("ln_bias")
            push!(initializers, _onnx_tensor_f32(scale_name, Int[size(scale)...], _row_major_f32_nd(scale)))
            push!(initializers, _onnx_tensor_f32(bias_name, Int[size(bias)...], _row_major_f32_nd(bias)))

            ln_out = next_name("ln_out")
            add_node!(
                "LayerNormalization",
                [current_name, scale_name, bias_name],
                [ln_out];
                attrs=[
                    _onnx_attr_int("axis", -norm_ndims),
                    _onnx_attr_float("epsilon", layer.eps),
                ],
                name_prefix="layernorm"
            )
            current_name = ln_out
        elseif layer isa MaxPool2d || layer isa AvgPool2d
            length(current_dims) == 4 || throw(ArgumentError("Pool export expects NHWC rank-4 input"))

            kernel = layer.kernel_size
            stride = layer.stride
            padding = layer.padding

            pool_in = nhwc_to_nchw(current_name)
            pool_out_nchw = next_name("pool_out")
            attrs = Vector{Vector{UInt8}}([
                _onnx_attr_ints("kernel_shape", [kernel[1], kernel[2]]),
                _onnx_attr_ints("strides", [stride[1], stride[2]]),
                _onnx_attr_ints("pads", [padding[1], padding[2], padding[1], padding[2]]),
            ])
            if layer isa AvgPool2d
                push!(attrs, _onnx_attr_int("count_include_pad", layer.count_include_pad ? 1 : 0))
            end

            add_node!(
                layer isa MaxPool2d ? "MaxPool" : "AveragePool",
                [pool_in],
                [pool_out_nchw];
                attrs=attrs,
                name_prefix="pool"
            )

            current_name = nchw_to_nhwc(pool_out_nchw)
            current_dims = Any[
                current_dims[1],
                _conv_pool_out_dim(current_dims[2], padding[1], kernel[1], stride[1]),
                _conv_pool_out_dim(current_dims[3], padding[2], kernel[2], stride[2]),
                current_dims[4],
            ]
        elseif layer isa AdaptiveAvgPool2d
            target = layer.output_size
            target == (1, 1) || throw(ArgumentError("AdaptiveAvgPool2d export currently supports only output_size=(1,1)"))
            length(current_dims) == 4 || throw(ArgumentError("AdaptiveAvgPool2d export expects NHWC rank-4 input"))

            gap_in = nhwc_to_nchw(current_name)
            gap_out = next_name("gap_out")
            add_node!("GlobalAveragePool", [gap_in], [gap_out]; name_prefix="globalavgpool")
            flat_out = next_name("gap_flat")
            add_node!("Flatten", [gap_out], [flat_out]; attrs=[_onnx_attr_int("axis", 1)], name_prefix="flatten")
            current_name = flat_out
            current_dims = Any[current_dims[1], current_dims[4]]
        elseif layer isa GlobalAvgPool
            length(current_dims) == 4 || throw(ArgumentError("GlobalAvgPool export expects NHWC rank-4 input"))
            gap_in = nhwc_to_nchw(current_name)
            gap_out = next_name("gap_out")
            add_node!("GlobalAveragePool", [gap_in], [gap_out]; name_prefix="globalavgpool")
            flat_out = next_name("gap_flat")
            add_node!("Flatten", [gap_out], [flat_out]; attrs=[_onnx_attr_int("axis", 1)], name_prefix="flatten")
            current_name = flat_out
            current_dims = Any[current_dims[1], current_dims[4]]
        elseif layer isa Flatten
            if length(current_dims) <= 2
                continue
            end
            axis = max(layer.start_dim - 1, 0)
            out_name = next_name("flatten")
            add_node!("Flatten", [current_name], [out_name]; attrs=[_onnx_attr_int("axis", axis)], name_prefix="flatten")
            leading = axis == 0 ? 1 : _shape_product(current_dims[1:axis])
            trailing = axis + 1 > length(current_dims) ? 1 : _shape_product(current_dims[axis+1:end])
            current_name = out_name
            current_dims = Any[leading, trailing]
        else
            op_type, attrs = _layer_activation_spec(layer)
            out_name = next_name("node_out")
            add_node!(op_type, [current_name], [out_name]; attrs=attrs, name_prefix="node")
            current_name = out_name
        end
    end

    if current_name != "output"
        push!(nodes, _onnx_node(
            "Identity";
            inputs=[current_name],
            outputs=["output"],
            name="output_identity"
        ))
        current_name = "output"
    end

    output_dims = current_dims
    graph = _onnx_graph(
        model_name,
        nodes,
        initializers,
        [_onnx_value_info("input", input_dims)],
        [_onnx_value_info(current_name, output_dims)]
    )
    _onnx_model(graph; opset=opset, producer_name=producer_name)
end

"""
    to_onnx(model, output_path; input_shape=nothing, opset=17, producer_name="Axiom.jl", model_name="AxiomModel")

Export a supported Axiom model to an ONNX binary file.

Current subset:
- `Sequential`/`Pipeline` models composed of:
  - `Dense`, `Conv2d`, `BatchNorm`, `LayerNorm`
  - `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` (currently `(1,1)`), `GlobalAvgPool`
  - activations (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`) and `Flatten`
"""
function to_onnx(
    model,
    output_path::AbstractString;
    input_shape = nothing,
    opset::Integer = 17,
    producer_name::AbstractString = "Axiom.jl",
    model_name::AbstractString = "AxiomModel"
)
    bytes = _to_onnx_bytes(
        model;
        input_shape=input_shape,
        opset=opset,
        producer_name=producer_name,
        model_name=model_name
    )
    open(String(output_path), "w") do io
        write(io, bytes)
    end
    String(output_path)
end

export_onnx(model, output_path::AbstractString; kwargs...) = to_onnx(model, output_path; kwargs...)
