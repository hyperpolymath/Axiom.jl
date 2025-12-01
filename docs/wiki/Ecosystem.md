# Ecosystem & Integrations

> Connect Axiom.jl to the wider Julia and ML ecosystem

## Julia Ecosystem

### Data Handling

#### MLDatasets.jl

Standard ML datasets with automatic download.

```julia
using Axiom
using MLDatasets

# MNIST
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# CIFAR-10
train_x, train_y = CIFAR10.traindata()

# ImageNet (requires download)
train_x, train_y = ImageNet.traindata()

# Text datasets
data = WikiText2.traindata()
```

#### DataFrames.jl

Tabular data processing.

```julia
using DataFrames, CSV

# Load CSV
df = CSV.read("data.csv", DataFrame)

# Prepare for Axiom
X = Matrix{Float32}(df[:, feature_cols])
y = Vector{Int}(df[:, :label])

# Create loader
loader = DataLoader((X, y), batch_size=32, shuffle=true)
```

#### Images.jl

Image processing pipeline.

```julia
using Images, ImageTransformations

# Load and preprocess
function load_image(path)
    img = load(path)
    img = imresize(img, (224, 224))
    img = channelview(img)  # CHW format
    img = permutedims(img, (2, 3, 1))  # HWC format
    return Float32.(img)
end

# Data augmentation
function augment(img)
    # Random horizontal flip
    if rand() > 0.5
        img = reverse(img, dims=2)
    end

    # Random rotation
    angle = randn() * 15  # ±15 degrees
    img = imrotate(img, deg2rad(angle))

    return img
end
```

### Visualization

#### Plots.jl / Makie.jl

Training visualization.

```julia
using Plots

# Training curves
plot(1:length(losses), losses,
    xlabel="Epoch",
    ylabel="Loss",
    title="Training Progress",
    legend=false)

# Confusion matrix
heatmap(confusion_matrix,
    xlabel="Predicted",
    ylabel="True",
    title="Confusion Matrix")

# Feature visualization
function visualize_filters(conv_layer)
    weights = conv_layer.weight  # (kH, kW, C_in, C_out)
    n_filters = size(weights, 4)

    plots = []
    for i in 1:min(n_filters, 16)
        push!(plots, heatmap(weights[:, :, 1, i], aspect_ratio=1))
    end

    plot(plots..., layout=(4, 4))
end
```

#### TensorBoardLogger.jl

Log to TensorBoard.

```julia
using TensorBoardLogger

logger = TBLogger("runs/experiment_1")

for epoch in 1:100
    # Training...

    # Log metrics
    log_value(logger, "loss/train", train_loss, step=epoch)
    log_value(logger, "loss/val", val_loss, step=epoch)
    log_value(logger, "accuracy", accuracy, step=epoch)

    # Log histograms
    log_histogram(logger, "weights/layer1", model.layers[1].weight, step=epoch)

    # Log images
    log_image(logger, "samples", sample_images, step=epoch)
end
```

### Automatic Differentiation

#### Zygote.jl

Automatic differentiation engine.

```julia
using Zygote

# Axiom uses Zygote internally
function loss(model, x, y)
    pred = forward(model, x)
    return cross_entropy(pred, y)
end

# Get gradients
grads = Zygote.gradient(params(model)) do
    loss(model, x, y)
end

# Or use Axiom's interface
grads = gradient(model, x, y, CrossEntropyLoss())
```

### GPU Computing

#### CUDA.jl

GPU acceleration (coming soon).

```julia
using CUDA

# Move to GPU
model_gpu = model |> gpu
x_gpu = x |> gpu

# Forward pass on GPU
y_gpu = forward(model_gpu, x_gpu)

# Back to CPU
y_cpu = y_gpu |> cpu
```

### Parallelism

#### Distributed.jl

Multi-process training.

```julia
using Distributed

# Add workers
addprocs(4)

@everywhere using Axiom

# Distributed data loader
@everywhere function load_shard(rank, world_size)
    # Load 1/world_size of data
end

# Synchronize gradients
function all_reduce_gradients!(grads)
    for g in grads
        g .= sum([@fetchfrom w g for w in workers()]) / nworkers()
    end
end
```

## External Integrations

### ONNX

Standard model interchange format.

```julia
using Axiom

# Export to ONNX
model = @axiom begin
    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)
end

export_onnx(model, "model.onnx",
    input_names=["input"],
    output_names=["output"],
    input_shapes=Dict("input" => (1, 784)),
    opset_version=13
)

# Import from ONNX
imported_model = load_onnx("external_model.onnx")
```

**ONNX Compatibility:**
| Layer Type | Export | Import |
|------------|--------|--------|
| Dense | ✓ | ✓ |
| Conv2D | ✓ | ✓ |
| MaxPool2D | ✓ | ✓ |
| BatchNorm | ✓ | ✓ |
| LayerNorm | ✓ | ✓ |
| ReLU, GELU, etc. | ✓ | ✓ |
| Softmax | ✓ | ✓ |

### PyTorch

Import PyTorch models.

```julia
using Axiom
using Axiom.PyTorchCompat

# Load .pt file
model = load_pytorch("model.pt")

# Load state dict
model = load_pytorch_state_dict("state_dict.pt", architecture)

# Convert specific layers
torch_layer = load_pytorch_layer("layer.pt")
axiom_layer = convert_layer(torch_layer)

# Weight mapping
weights = load_pytorch_weights("model.pt")
copy_weights!(axiom_model, weights,
    mapping=Dict(
        "fc1.weight" => "layers.1.weight",
        "fc1.bias" => "layers.1.bias"
    )
)
```

### HuggingFace

Load pretrained transformers.

```julia
using Axiom
using Axiom.HuggingFaceCompat

# Load pretrained model
model = from_pretrained("bert-base-uncased")

# Load tokenizer
tokenizer = load_tokenizer("bert-base-uncased")

# Inference
text = "Hello, world!"
tokens = tokenize(tokenizer, text)
output = forward(model, tokens)
```

### MLflow

Experiment tracking.

```julia
using MLflow

# Start run
MLflow.start_run(experiment_name="axiom_experiment")

# Log parameters
MLflow.log_param("learning_rate", 0.001)
MLflow.log_param("batch_size", 32)
MLflow.log_param("epochs", 100)

# Training loop
for epoch in 1:100
    # Train...

    # Log metrics
    MLflow.log_metric("loss", loss, step=epoch)
    MLflow.log_metric("accuracy", accuracy, step=epoch)
end

# Log model
save_model(model, "model.axiom")
MLflow.log_artifact("model.axiom")

# Log verification certificate
save_certificate(cert, "cert.json")
MLflow.log_artifact("cert.json")

MLflow.end_run()
```

### Weights & Biases

Experiment tracking with W&B.

```julia
using WandB

# Initialize
wandb = WandB.init(
    project="axiom-experiments",
    config=Dict(
        "learning_rate" => 0.001,
        "architecture" => "MLP",
        "dataset" => "MNIST"
    )
)

# Log metrics
for epoch in 1:100
    # Train...

    WandB.log(Dict(
        "epoch" => epoch,
        "loss" => loss,
        "accuracy" => accuracy
    ))
end

# Log model
WandB.save("model.axiom")

WandB.finish()
```

## Deployment Integrations

### Docker

Containerized deployment.

```dockerfile
# Dockerfile
FROM julia:1.9

# Install Axiom
RUN julia -e 'using Pkg; Pkg.add("Axiom")'

# Copy model
COPY model.axiom /app/model.axiom

# Copy inference script
COPY serve.jl /app/serve.jl

WORKDIR /app
EXPOSE 8080

CMD ["julia", "serve.jl"]
```

```julia
# serve.jl
using Axiom
using HTTP, JSON

model = load_model("model.axiom")

function handle_request(req)
    data = JSON.parse(String(req.body))
    input = Float32.(data["input"])

    output = forward(model, reshape(input, 1, :))

    return HTTP.Response(200,
        JSON.json(Dict("prediction" => vec(output)))
    )
end

HTTP.serve(handle_request, "0.0.0.0", 8080)
```

### gRPC

High-performance RPC.

```julia
using gRPC

# Define service
service = AxiomInference(model)

@grpc function predict(request::PredictRequest)::PredictResponse
    input = reshape(request.input, 1, :)
    output = forward(model, input)
    return PredictResponse(prediction=vec(output))
end

# Serve
serve(service, "0.0.0.0:50051")
```

### REST API

HTTP inference endpoint.

```julia
using Genie, JSON

model = load_model("model.axiom")

route("/predict", method=POST) do
    data = jsonpayload()
    input = Float32.(data["input"])

    output = forward(model, reshape(input, 1, :))

    return json(Dict(
        "prediction" => vec(output),
        "model_version" => model.version
    ))
end

route("/health") do
    return json(Dict("status" => "healthy"))
end

up(8080)
```

## Cloud Integrations

### AWS

```julia
using AWS

# S3 for model storage
s3_put("s3://models/axiom/model.axiom", read("model.axiom"))

# Load from S3
model_bytes = s3_get("s3://models/axiom/model.axiom")
model = load_model(IOBuffer(model_bytes))

# SageMaker deployment (via container)
# See Docker integration above
```

### Google Cloud

```julia
using GoogleCloud

# Cloud Storage
upload_object("gs://models/axiom/model.axiom", "model.axiom")

# Download
download_object("gs://models/axiom/model.axiom", "local_model.axiom")

# Vertex AI (via container)
# See Docker integration above
```

### Azure

```julia
using Azure

# Blob storage
upload_blob("models", "axiom/model.axiom", "model.axiom")

# Azure ML deployment
# Via container or Azure ML SDK
```

## Interoperability Matrix

| Integration | Status | Notes |
|-------------|--------|-------|
| **Data** | | |
| MLDatasets.jl | ✓ Stable | Full support |
| DataFrames.jl | ✓ Stable | Full support |
| Images.jl | ✓ Stable | Full support |
| **Visualization** | | |
| Plots.jl | ✓ Stable | Full support |
| Makie.jl | ✓ Stable | Full support |
| TensorBoard | ✓ Stable | Via TensorBoardLogger.jl |
| **Compute** | | |
| CUDA.jl | ⚠ Beta | GPU support in development |
| Distributed.jl | ⚠ Beta | Multi-node training |
| **Model Exchange** | | |
| ONNX | ✓ Stable | Import/Export |
| PyTorch | ✓ Stable | Import only |
| TensorFlow | ⚠ Beta | Via ONNX |
| HuggingFace | ⚠ Beta | Selected models |
| **Tracking** | | |
| MLflow | ✓ Stable | Full support |
| W&B | ✓ Stable | Full support |
| **Deployment** | | |
| Docker | ✓ Stable | Full support |
| REST/gRPC | ✓ Stable | Full support |
| AWS | ✓ Stable | S3, SageMaker |
| GCP | ✓ Stable | GCS, Vertex AI |
| Azure | ✓ Stable | Blob, Azure ML |

---

*Next: [Deployment](Deployment.md) for production deployment strategies*
