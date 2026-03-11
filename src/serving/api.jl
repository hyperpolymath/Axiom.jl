# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl API Serving Utilities
#
# Lightweight serving surfaces for REST and GraphQL, plus gRPC proto/message helpers.

function _to_matrix_input(input)
    if input isa AbstractMatrix
        return Float32.(input)
    end

    if input isa AbstractVector{<:Number}
        values = Float32.(collect(input))
        return reshape(values, 1, :)
    end

    if input isa AbstractVector
        rows = Vector{Vector{Float32}}()
        for row in input
            row isa AbstractVector || throw(ArgumentError("Each batch item must be a vector"))
            push!(rows, Float32.(collect(row)))
        end

        isempty(rows) && throw(ArgumentError("Input batch must not be empty"))
        width = length(rows[1])
        all(length(row) == width for row in rows) || throw(ArgumentError("All rows must have the same width"))

        matrix = Matrix{Float32}(undef, length(rows), width)
        for (i, row) in enumerate(rows)
            matrix[i, :] = row
        end
        return matrix
    end

    throw(ArgumentError("Unsupported input format; expected vector, matrix, or vector-of-vectors"))
end

function _to_json_output(y)
    data = y isa AbstractTensor ? y.data : y

    if data isa AbstractMatrix
        return [collect(data[i, :]) for i in 1:size(data, 1)]
    end
    if data isa AbstractVector
        return collect(data)
    end
    data
end

function _predict_output(model, input)
    x = _to_matrix_input(input)
    y = model(x)
    _to_json_output(y)
end

function _json_response(
    status::Int,
    payload::Dict{String, Any};
    content_type::AbstractString = "application/json"
)
    body = JSON.json(payload)
    headers = ["Content-Type" => String(content_type)]
    HTTP.Response(status, headers, body)
end

function _request_path(req::HTTP.Request)
    target = String(req.target)
    try
        uri = HTTP.URI(target)
        isempty(uri.path) ? "/" : uri.path
    catch
        target
    end
end

function _rest_handler(model, req::HTTP.Request, predict_route::String, health_route::String)
    method = String(req.method)
    path = _request_path(req)

    if method == "GET" && path == health_route
        return _json_response(200, Dict{String, Any}("status" => "ok"))
    end

    if method == "POST" && path == predict_route
        try
            payload = JSON.parse(String(req.body))
            haskey(payload, "input") || return _json_response(400, Dict{String, Any}("error" => "Missing `input` field"))
            output = _predict_output(model, payload["input"])
            return _json_response(200, Dict{String, Any}("output" => output))
        catch e
            return _json_response(400, Dict{String, Any}("error" => sprint(showerror, e)))
        end
    end

    _json_response(404, Dict{String, Any}("error" => "Not found"))
end

"""
    serve_rest(model; host="127.0.0.1", port=8080, predict_route="/predict", health_route="/health", background=false)

Start a REST server exposing `POST /predict` and `GET /health`.
Returns an `HTTP.Server` object when `background=true`.
"""
function serve_rest(
    model;
    host::AbstractString = "127.0.0.1",
    port::Integer = 8080,
    predict_route::AbstractString = "/predict",
    health_route::AbstractString = "/health",
    background::Bool = false
)
    handler = req -> _rest_handler(model, req, String(predict_route), String(health_route))
    if background
        return HTTP.serve!(handler, String(host), Int(port); verbose=false)
    end
    HTTP.serve(handler, String(host), Int(port); verbose=false)
end

"""
    graphql_execute(model, query; variables=Dict{String,Any}())

Execute a minimal GraphQL operation set:
- `health`
- `predict` (requires `variables["input"]`)
"""
function graphql_execute(model, query::AbstractString; variables=Dict{String, Any}())
    q = lowercase(strip(String(query)))

    if occursin("health", q)
        return Dict{String, Any}("data" => Dict{String, Any}("health" => "ok"))
    end

    if occursin("predict", q)
        input = get(variables, "input", get(variables, :input, nothing))
        input === nothing && return Dict{String, Any}("errors" => [Dict{String, Any}("message" => "predict requires variables.input")])
        output = _predict_output(model, input)
        return Dict{String, Any}("data" => Dict{String, Any}("predict" => output))
    end

    Dict{String, Any}("errors" => [Dict{String, Any}("message" => "Unsupported GraphQL operation")])
end

function _graphql_handler(model, req::HTTP.Request, graphql_route::String, health_route::String)
    method = String(req.method)
    path = _request_path(req)

    if method == "GET" && path == health_route
        return _json_response(200, Dict{String, Any}("status" => "ok"))
    end

    if method == "POST" && path == graphql_route
        try
            payload = JSON.parse(String(req.body))
            query = String(get(payload, "query", ""))
            variables = Dict{String, Any}()
            if haskey(payload, "variables") && payload["variables"] isa AbstractDict
                for (k, v) in payload["variables"]
                    variables[String(k)] = v
                end
            end
            if isempty(variables) && haskey(payload, "input")
                variables["input"] = payload["input"]
            end

            result = graphql_execute(model, query; variables=variables)
            status = haskey(result, "errors") ? 400 : 200
            return _json_response(status, result)
        catch e
            return _json_response(400, Dict{String, Any}("errors" => [Dict{String, Any}("message" => sprint(showerror, e))]))
        end
    end

    _json_response(404, Dict{String, Any}("error" => "Not found"))
end

"""
    serve_graphql(model; host="127.0.0.1", port=8081, graphql_route="/graphql", health_route="/health", background=false)

Start a GraphQL server with a minimal operation set (`health`, `predict`).
Returns an `HTTP.Server` object when `background=true`.
"""
function serve_graphql(
    model;
    host::AbstractString = "127.0.0.1",
    port::Integer = 8081,
    graphql_route::AbstractString = "/graphql",
    health_route::AbstractString = "/health",
    background::Bool = false
)
    handler = req -> _graphql_handler(model, req, String(graphql_route), String(health_route))
    if background
        return HTTP.serve!(handler, String(host), Int(port); verbose=false)
    end
    HTTP.serve(handler, String(host), Int(port); verbose=false)
end

function _grpc_paths(package_name::String, service_name::String)
    base = "/$(package_name).$(service_name)"
    (
        predict = "$(base)/Predict",
        health = "$(base)/Health",
    )
end

const _GRPC_PB_WIRE_VARINT = 0
const _GRPC_PB_WIRE_LEN = 2
const _GRPC_PB_WIRE_32 = 5

function _grpc_pb_read_varint(data::Vector{UInt8}, idx::Int)
    result = UInt64(0)
    shift = 0
    n = length(data)
    while true
        idx <= n || throw(ArgumentError("Malformed protobuf varint"))
        byte = data[idx]
        idx += 1
        result |= (UInt64(byte & 0x7f) << shift)
        if (byte & 0x80) == 0
            return result, idx
        end
        shift += 7
        shift < 64 || throw(ArgumentError("Malformed protobuf varint (overflow)"))
    end
end

function _grpc_pb_skip_field(data::Vector{UInt8}, idx::Int, wire::Int)
    n = length(data)
    if wire == _GRPC_PB_WIRE_VARINT
        _, idx = _grpc_pb_read_varint(data, idx)
        return idx
    elseif wire == _GRPC_PB_WIRE_LEN
        len, idx = _grpc_pb_read_varint(data, idx)
        stop_idx = idx + Int(len)
        stop_idx - 1 <= n || throw(ArgumentError("Malformed length-delimited protobuf field"))
        return stop_idx
    elseif wire == 1 # 64-bit
        stop_idx = idx + 8
        stop_idx - 1 <= n || throw(ArgumentError("Malformed 64-bit protobuf field"))
        return stop_idx
    elseif wire == _GRPC_PB_WIRE_32
        stop_idx = idx + 4
        stop_idx - 1 <= n || throw(ArgumentError("Malformed 32-bit protobuf field"))
        return stop_idx
    end
    throw(ArgumentError("Unsupported protobuf wire type: $wire"))
end

function _grpc_le_f32(data::Vector{UInt8}, idx::Int)
    idx + 3 <= length(data) || throw(ArgumentError("Malformed float32 payload"))
    u = UInt32(data[idx]) |
        (UInt32(data[idx + 1]) << 8) |
        (UInt32(data[idx + 2]) << 16) |
        (UInt32(data[idx + 3]) << 24)
    reinterpret(Float32, u)
end

function _grpc_pb_decode_predict_request(payload::Vector{UInt8})
    values = Float32[]
    idx = 1
    n = length(payload)

    while idx <= n
        key, idx = _grpc_pb_read_varint(payload, idx)
        field = Int(key >> 3)
        wire = Int(key & 0x07)

        if field == 1
            if wire == _GRPC_PB_WIRE_LEN
                block_len, idx = _grpc_pb_read_varint(payload, idx)
                stop_idx = idx + Int(block_len) - 1
                stop_idx <= n || throw(ArgumentError("Malformed packed float payload"))
                while idx <= stop_idx
                    push!(values, _grpc_le_f32(payload, idx))
                    idx += 4
                end
            elseif wire == _GRPC_PB_WIRE_32
                push!(values, _grpc_le_f32(payload, idx))
                idx += 4
            else
                throw(ArgumentError("Unexpected wire type for PredictRequest.input: $wire"))
            end
        else
            idx = _grpc_pb_skip_field(payload, idx, wire)
        end
    end

    values
end

function _grpc_pb_decode_health_request(payload::Vector{UInt8})
    idx = 1
    n = length(payload)
    while idx <= n
        key, idx = _grpc_pb_read_varint(payload, idx)
        wire = Int(key & 0x07)
        idx = _grpc_pb_skip_field(payload, idx, wire)
    end
    nothing
end

function _grpc_pb_varint(value::Integer)
    u = value >= 0 ? UInt64(value) : reinterpret(UInt64, Int64(value))
    out = UInt8[]
    while u >= 0x80
        push!(out, UInt8((u & 0x7f) | 0x80))
        u >>= 7
    end
    push!(out, UInt8(u))
    out
end

function _grpc_pb_write_key(field::Integer, wire::Integer)
    _grpc_pb_varint((UInt64(field) << 3) | UInt64(wire))
end

function _grpc_pb_encode_float_list_field1(values::Vector{Float32})
    payload = collect(reinterpret(UInt8, values))
    out = UInt8[]
    append!(out, _grpc_pb_write_key(1, _GRPC_PB_WIRE_LEN))
    append!(out, _grpc_pb_varint(length(payload)))
    append!(out, payload)
    out
end

function _grpc_pb_encode_health_response(status::AbstractString)
    body = Vector{UInt8}(codeunits(String(status)))
    out = UInt8[]
    append!(out, _grpc_pb_write_key(1, _GRPC_PB_WIRE_LEN))
    append!(out, _grpc_pb_varint(length(body)))
    append!(out, body)
    out
end

function _grpc_collect_numbers!(out::Vector{Float32}, value)
    if value isa Number
        push!(out, Float32(value))
        return
    end
    value isa AbstractArray || throw(ArgumentError("gRPC binary response requires numeric arrays"))
    for v in value
        _grpc_collect_numbers!(out, v)
    end
end

function _grpc_flatten_numbers(value)
    out = Float32[]
    _grpc_collect_numbers!(out, value)
    out
end

function _grpc_encode_frame(payload::Vector{UInt8}; compressed::Bool = false)
    n = length(payload)
    out = UInt8[compressed ? 0x01 : 0x00]
    push!(out, UInt8((n >> 24) & 0xff))
    push!(out, UInt8((n >> 16) & 0xff))
    push!(out, UInt8((n >> 8) & 0xff))
    push!(out, UInt8(n & 0xff))
    append!(out, payload)
    out
end

function _grpc_decode_frame(body::Vector{UInt8})
    length(body) >= 5 || throw(ArgumentError("gRPC body must include 5-byte frame header"))
    compressed = body[1]
    compressed == 0x00 || throw(ArgumentError("Compressed gRPC frames are not supported"))
    msg_len = (Int(body[2]) << 24) | (Int(body[3]) << 16) | (Int(body[4]) << 8) | Int(body[5])
    msg_len >= 0 || throw(ArgumentError("Invalid gRPC frame length"))
    length(body) == 5 + msg_len || throw(ArgumentError("gRPC frame length mismatch"))
    body[6:end]
end

function _grpc_binary_response(payload::Vector{UInt8}; status::Int = 0, message::AbstractString = "")
    headers = [
        "Content-Type" => "application/grpc",
        "grpc-status" => string(status),
    ]
    isempty(message) || push!(headers, "grpc-message" => String(message))
    HTTP.Response(200, headers, payload)
end

function _grpc_binary_predict(model, frame_payload::Vector{UInt8})
    request_input = _grpc_pb_decode_predict_request(frame_payload)
    output = _predict_output(model, request_input)
    response = _grpc_pb_encode_float_list_field1(_grpc_flatten_numbers(output))
    _grpc_binary_response(_grpc_encode_frame(response))
end

function _grpc_binary_health(frame_payload::Vector{UInt8})
    _grpc_pb_decode_health_request(frame_payload)
    response = _grpc_pb_encode_health_response("SERVING")
    _grpc_binary_response(_grpc_encode_frame(response))
end

function _grpc_binary_error(message::AbstractString; code::Int = 13)
    _grpc_binary_response(_grpc_encode_frame(UInt8[]); status=code, message=message)
end

function _grpc_handler(
    model,
    req::HTTP.Request,
    package_name::String,
    service_name::String
)
    method = String(req.method)
    path = _request_path(req)
    paths = _grpc_paths(package_name, service_name)
    content_type = lowercase(String(HTTP.header(req, "Content-Type", "")))

    if startswith(content_type, "application/grpc") && !occursin("+json", content_type)
        method == "POST" || return _grpc_binary_error("gRPC binary endpoints require POST"; code=12)
        try
            frame_payload = _grpc_decode_frame(collect(req.body))
            if path == paths.predict
                return _grpc_binary_predict(model, frame_payload)
            elseif path == paths.health
                return _grpc_binary_health(frame_payload)
            end
            return _grpc_binary_error("Unknown gRPC method path: $(path)"; code=12)
        catch e
            return _grpc_binary_error(sprint(showerror, e); code=13)
        end
    end

    if (method == "POST" || method == "GET") && path == paths.health
        return _json_response(200, grpc_health(); content_type="application/grpc+json")
    end

    if method == "POST" && path == paths.predict
        try
            payload = JSON.parse(String(req.body))
            haskey(payload, "input") || return _json_response(
                400,
                Dict{String, Any}("error" => "Missing `input` field");
                content_type="application/grpc+json"
            )
            return _json_response(
                200,
                grpc_predict(model, payload["input"]);
                content_type="application/grpc+json"
            )
        catch e
            return _json_response(
                400,
                Dict{String, Any}("error" => sprint(showerror, e));
                content_type="application/grpc+json"
            )
        end
    end

    _json_response(404, Dict{String, Any}("error" => "Not found"); content_type="application/grpc+json")
end

"""
    serve_grpc(model; host="127.0.0.1", port=50051, package_name="axiom.v1", service_name="AxiomInference", background=false)

Start an in-tree network gRPC bridge that serves:
- `POST /{package}.{service}/Predict`
- `POST|GET /{package}.{service}/Health`

Request/response modes:
- `application/grpc`: unary protobuf frame handling for `Predict` and `Health`
- `application/grpc+json`: JSON bridge payloads with generated proto field names

Returns an `HTTP.Server` object when `background=true`.
"""
function serve_grpc(
    model;
    host::AbstractString = "127.0.0.1",
    port::Integer = 50051,
    package_name::AbstractString = "axiom.v1",
    service_name::AbstractString = "AxiomInference",
    background::Bool = false
)
    handler = req -> _grpc_handler(model, req, String(package_name), String(service_name))
    if background
        return HTTP.serve!(handler, String(host), Int(port); verbose=false)
    end
    HTTP.serve(handler, String(host), Int(port); verbose=false)
end

"""
    generate_grpc_proto([output_path]; package_name="axiom.v1", service_name="AxiomInference")

Generate a gRPC `.proto` contract for inference services.
"""
function generate_grpc_proto(
    output_path::AbstractString = "axiom_inference.proto";
    package_name::AbstractString = "axiom.v1",
    service_name::AbstractString = "AxiomInference"
)
    proto = """
syntax = "proto3";

package $(package_name);

service $(service_name) {
  rpc Predict (PredictRequest) returns (PredictResponse);
  rpc Health (HealthRequest) returns (HealthResponse);
}

message PredictRequest {
  repeated float input = 1;
}

message PredictResponse {
  repeated float output = 1;
}

message HealthRequest {}

message HealthResponse {
  string status = 1;
}
"""
    open(String(output_path), "w") do io
        write(io, proto)
    end
    @info "gRPC proto generated at $(output_path)"
    String(output_path)
end

"""
    grpc_predict(model, input)

In-process gRPC-style predict handler for generated `PredictRequest` payload data.
"""
grpc_predict(model, input) = Dict{String, Any}("output" => _predict_output(model, input))

"""
    grpc_health()

In-process gRPC-style health handler.
"""
grpc_health() = Dict{String, Any}("status" => "SERVING")

"""
    grpc_support_status()

Describe currently shipped gRPC support components.
"""
function grpc_support_status()
    Dict{String, Any}(
        "proto_generation" => true,
        "inprocess_handlers" => true,
        "network_server" => true,
        "binary_wire" => true,
        "network_mode" => "in-tree server with gRPC method paths and unary protobuf frame handling",
        "note" => "Use `serve_grpc(...)` for in-tree serving (`application/grpc` or `application/grpc+json`), or generate proto contracts for external runtimes."
    )
end
