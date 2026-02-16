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

function _json_response(status::Int, payload::Dict{String, Any})
    body = JSON.json(payload)
    headers = ["Content-Type" => "application/json"]
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
        "network_server" => false,
        "note" => "Use the generated proto with your preferred gRPC runtime to expose network gRPC endpoints."
    )
end
