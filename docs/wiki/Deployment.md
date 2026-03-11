<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# Deployment Guide

> From development to production: deploy Axiom.jl models anywhere

## Deployment Options

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Axiom.jl Deployment                             │
├──────────────┬──────────────┬──────────────┬───────────────────────┤
│    Server    │    Edge      │   Browser    │      Mobile           │
├──────────────┼──────────────┼──────────────┼───────────────────────┤
│  REST/gRPC   │  Zig/Julia   │  WASM*       │  JNI/Swift*           │
│  Docker      │  Embedded    │              │                       │
│  Kubernetes  │  RTOS*       │              │                       │
└──────────────┴──────────────┴──────────────┴───────────────────────┘
                        * Coming soon
```

## Server Deployment

### Option 1: Julia Native Server

The simplest approach - run Julia directly.

```julia
# server.jl
using Axiom
using HTTP
using JSON

# Load model once at startup
const MODEL = load_model("model.axiom")
const CERT = load_certificate("model.cert")

# Validation
function validate_input(data)
    haskey(data, "input") || error("Missing 'input' field")
    input = data["input"]
    length(input) == 784 || error("Input must be 784-dimensional")
    all(0 .<= input .<= 1) || error("Input must be in [0, 1]")
    return Float32.(input)
end

# Inference endpoint
function handle_predict(req::HTTP.Request)
    try
        data = JSON.parse(String(req.body))
        input = validate_input(data)

        # Reshape and predict
        x = reshape(input, 1, :)
        output = forward(MODEL, x)

        return HTTP.Response(200,
            ["Content-Type" => "application/json"],
            JSON.json(Dict(
                "prediction" => vec(output),
                "confidence" => maximum(output),
                "class" => argmax(vec(output)) - 1,
                "model_hash" => bytes2hex(CERT.model_hash)[1:16],
                "verified" => true
            ))
        )
    catch e
        return HTTP.Response(400,
            ["Content-Type" => "application/json"],
            JSON.json(Dict("error" => string(e)))
        )
    end
end

# Health check
function handle_health(req::HTTP.Request)
    return HTTP.Response(200,
        ["Content-Type" => "application/json"],
        JSON.json(Dict(
            "status" => "healthy",
            "model_loaded" => true,
            "certificate_valid" => validate_certificate(CERT)
        ))
    )
end

# Router
function router(req::HTTP.Request)
    if req.target == "/predict" && req.method == "POST"
        return handle_predict(req)
    elseif req.target == "/health" && req.method == "GET"
        return handle_health(req)
    else
        return HTTP.Response(404, "Not Found")
    end
end

# Start server
println("Starting Axiom inference server on port 8080...")
HTTP.serve(router, "0.0.0.0", 8080)
```

**Run:**
```bash
julia --project -t auto server.jl
```

### Option 2: Docker Container

Production-ready containerization.

```dockerfile
# Dockerfile
FROM julia:1.9-bullseye

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY Project.toml Manifest.toml ./

# Install Julia packages
RUN julia --project -e 'using Pkg; Pkg.instantiate()'

# Copy source code
COPY src/ ./src/

# Copy model and certificate
COPY model.axiom cert.json ./

# Copy server script
COPY server.jl ./

# Precompile
RUN julia --project -e 'using Axiom'

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD ["julia", "--project", "-t", "auto", "server.jl"]
```

**Build and run:**
```bash
docker build -t axiom-inference .
docker run -p 8080:8080 axiom-inference
```

### Option 3: Kubernetes

Scale horizontally with K8s.

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: axiom-inference
  labels:
    app: axiom-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: axiom-inference
  template:
    metadata:
      labels:
        app: axiom-inference
    spec:
      containers:
      - name: axiom
        image: your-registry/axiom-inference:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: JULIA_NUM_THREADS
          value: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: axiom-inference
spec:
  selector:
    app: axiom-inference
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: axiom-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: axiom-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Deploy:**
```bash
kubectl apply -f deployment.yaml
```

## Edge Deployment

### Edge Runtime Path

Current supported path for edge is Julia + optional Zig backend library.

```julia
using Axiom

model = load_model("model.axiom")

# Optional: compile against Zig backend when lib is available
model_runtime = compile(
    model,
    backend = ZigBackend("/path/to/libaxiom_zig.so"),
    verify = false,
    optimize = :none,
)
```

For constrained targets, package the runtime with precompiled artifacts and
use CPU fallback if accelerator/runtime dependencies are unavailable.

### Embedded C Interface

For C/C++ integration:

```c
// axiom.h
#ifndef AXIOM_H
#define AXIOM_H

#include <stddef.h>

// Initialize model (call once)
int axiom_init(const char* model_path);

// Run inference
// input: array of float32, size input_size
// output: array of float32, size output_size
int axiom_predict(const float* input, size_t input_size,
                  float* output, size_t output_size);

// Cleanup
void axiom_cleanup(void);

#endif
```

**Usage in C:**
```c
#include "axiom.h"
#include <stdio.h>

int main() {
    if (axiom_init("model.bin") != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    float input[784] = { /* ... */ };
    float output[10];

    axiom_predict(input, 784, output, 10);

    // Find max class
    int max_class = 0;
    for (int i = 1; i < 10; i++) {
        if (output[i] > output[max_class]) {
            max_class = i;
        }
    }

    printf("Predicted class: %d\n", max_class);

    axiom_cleanup();
    return 0;
}
```

### Raspberry Pi Deployment

Step-by-step for Raspberry Pi 4:

```bash
# On development machine: prepare model/runtime artifacts
scp model.bin pi@raspberrypi:~/
# Optional Zig backend shared library for ARM target:
# scp libaxiom_zig.so pi@raspberrypi:~/

# On Raspberry Pi
ssh pi@raspberrypi

# Install Julia (ARM64)
curl -fsSL https://julialang-s3.julialang.org/bin/linux/aarch64/1.9/julia-1.9.0-linux-aarch64.tar.gz | tar -xz
export PATH="$PATH:$PWD/julia-1.9.0/bin"

# Run inference
julia -e '
using Axiom
model = load_model("model.bin")
x = rand(Float32, 1, 784)
@time y = forward(model, x)  # ~5ms on Pi 4
'
```

## Cloud Deployment

### AWS SageMaker

```python
# sagemaker_deploy.py
import sagemaker
from sagemaker.model import Model

# Create model
model = Model(
    image_uri="your-account.dkr.ecr.region.amazonaws.com/axiom-inference:latest",
    role="arn:aws:iam::your-account:role/SageMakerRole",
    model_data="s3://your-bucket/model.tar.gz"
)

# Deploy endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.c5.xlarge",
    endpoint_name="axiom-endpoint"
)

# Invoke
result = predictor.predict({"input": [0.0] * 784})
```

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/your-project/axiom-inference

# Deploy
gcloud run deploy axiom-inference \
    --image gcr.io/your-project/axiom-inference \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10
```

### Azure Container Instances

```bash
# Create container
az container create \
    --resource-group axiom-rg \
    --name axiom-inference \
    --image your-registry.azurecr.io/axiom-inference:latest \
    --cpu 2 \
    --memory 2 \
    --ports 8080 \
    --dns-name-label axiom-inference
```

## Model Versioning

### Semantic Versioning

```julia
# Version your models
model = @axiom version="1.2.3" begin
    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)
end

# Save with version
save_model(model, "model_v1.2.3.axiom")
```

### Model Registry

```julia
# model_registry.jl
struct ModelRegistry
    storage_path::String
    models::Dict{String, Vector{ModelVersion}}
end

struct ModelVersion
    version::VersionNumber
    path::String
    certificate_path::String
    created_at::DateTime
    metrics::Dict{String, Float64}
end

function register_model!(registry, name, model, cert; metrics=Dict())
    version = ModelVersion(
        model.version,
        joinpath(registry.storage_path, "$name-$(model.version).axiom"),
        joinpath(registry.storage_path, "$name-$(model.version).cert"),
        now(),
        metrics
    )

    save_model(model, version.path)
    save_certificate(cert, version.certificate_path)

    if !haskey(registry.models, name)
        registry.models[name] = ModelVersion[]
    end
    push!(registry.models[name], version)
end

function get_latest(registry, name)
    versions = registry.models[name]
    return versions[end]
end

function get_version(registry, name, version)
    versions = registry.models[name]
    idx = findfirst(v -> v.version == version, versions)
    return versions[idx]
end
```

## Monitoring

### Metrics Collection

```julia
# metrics.jl
using Prometheus

# Define metrics
const INFERENCE_LATENCY = Histogram(
    "axiom_inference_latency_seconds",
    "Inference latency in seconds",
    ["model", "version"]
)

const INFERENCE_COUNT = Counter(
    "axiom_inference_total",
    "Total inference requests",
    ["model", "version", "status"]
)

const MODEL_CONFIDENCE = Histogram(
    "axiom_model_confidence",
    "Model prediction confidence",
    ["model", "class"]
)

# Instrumented inference
function monitored_predict(model, x)
    start = time()

    try
        output = forward(model, x)

        # Record metrics
        observe!(INFERENCE_LATENCY, time() - start;
            labels=Dict("model" => model.name, "version" => string(model.version)))

        inc!(INFERENCE_COUNT;
            labels=Dict("model" => model.name, "version" => string(model.version), "status" => "success"))

        confidence = maximum(output)
        predicted_class = argmax(output)
        observe!(MODEL_CONFIDENCE, confidence;
            labels=Dict("model" => model.name, "class" => string(predicted_class)))

        return output

    catch e
        inc!(INFERENCE_COUNT;
            labels=Dict("model" => model.name, "version" => string(model.version), "status" => "error"))
        rethrow(e)
    end
end
```

### Alerting

```yaml
# prometheus_rules.yaml
groups:
- name: axiom-alerts
  rules:
  - alert: HighInferenceLatency
    expr: histogram_quantile(0.99, axiom_inference_latency_seconds_bucket) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High inference latency detected"

  - alert: LowModelConfidence
    expr: histogram_quantile(0.5, axiom_model_confidence_bucket) < 0.7
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Model confidence dropping"

  - alert: InferenceErrors
    expr: rate(axiom_inference_total{status="error"}[5m]) > 0.01
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Inference error rate elevated"
```

## A/B Testing

```julia
# ab_testing.jl
struct ABTest
    name::String
    models::Dict{Symbol, Any}  # :control, :treatment
    traffic_split::Float64     # Fraction to treatment
end

function route_request(test::ABTest, request_id::String)
    # Deterministic routing based on request ID
    hash_val = hash(request_id)
    if (hash_val % 100) / 100 < test.traffic_split
        return :treatment
    else
        return :control
    end
end

function ab_predict(test::ABTest, x, request_id::String)
    variant = route_request(test, request_id)
    model = test.models[variant]

    output = forward(model, x)

    # Log for analysis
    log_ab_event(test.name, variant, request_id, maximum(output))

    return output, variant
end

# Setup A/B test
test = ABTest(
    "new_model_v2",
    Dict(
        :control => load_model("model_v1.axiom"),
        :treatment => load_model("model_v2.axiom")
    ),
    0.1  # 10% to new model
)
```

## Rollback Strategy

```julia
# rollback.jl
struct DeploymentManager
    registry::ModelRegistry
    current::Ref{ModelVersion}
    previous::Ref{Union{ModelVersion, Nothing}}
end

function deploy!(manager::DeploymentManager, name, version)
    model_version = get_version(manager.registry, name, version)

    # Validate certificate
    cert = load_certificate(model_version.certificate_path)
    if !validate_certificate(cert)
        error("Invalid certificate for version $version")
    end

    # Save current as previous
    manager.previous[] = manager.current[]

    # Deploy new version
    manager.current[] = model_version

    println("Deployed $name v$version")
end

function rollback!(manager::DeploymentManager)
    if isnothing(manager.previous[])
        error("No previous version to rollback to")
    end

    manager.current[], manager.previous[] = manager.previous[], manager.current[]

    println("Rolled back to v$(manager.current[].version)")
end
```

## Production Checklist

### Pre-Deployment
- [ ] Model trained and validated
- [ ] Verification certificate generated
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Performance benchmarks acceptable
- [ ] Security review completed

### Deployment
- [ ] Docker image built and tested
- [ ] Kubernetes manifests validated
- [ ] Health checks configured
- [ ] Resource limits set
- [ ] Autoscaling configured

### Post-Deployment
- [ ] Monitoring dashboards active
- [ ] Alerting rules configured
- [ ] Logging enabled
- [ ] Rollback procedure tested
- [ ] A/B testing framework ready

---

*Questions? See our [FAQ](FAQ.md) or join [Discord](https://discord.gg/axiom-jl)*
