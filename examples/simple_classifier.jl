# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Simple Classifier Example
#
# Demonstrates the core features of Axiom.jl

using Axiom

println("═" ^ 60)
println("  Axiom.jl - Provably Correct Machine Learning")
println("═" ^ 60)

# ==============================================================================
# 1. Define a Model using @axiom DSL
# ==============================================================================

println("\n1. Defining model with @axiom DSL...")

@axiom BinaryClassifier begin
    input :: Tensor{Float32, (:batch, 20)}
    output :: Tensor{Float32, (:batch, 2)}

    # Simple 2-layer network
    hidden = input |> Dense(20, 32, relu)
    logits = hidden |> Dense(32, 2)
    output = logits |> Softmax

    # Formal guarantees
    @ensure sum(output) ≈ 1.0  # Valid probabilities
end

println("   ✓ Model defined with compile-time guarantees")

# ==============================================================================
# 2. Create Model Instance
# ==============================================================================

println("\n2. Creating model instances...")

# Using @axiom-defined model
model1 = BinaryClassifier()
println("   ✓ @axiom model created")

# Using Sequential builder
model2 = Sequential(
    Dense(20, 32, relu),
    Dense(32, 2),
    Softmax()
)
println("   ✓ Sequential model created")

# ==============================================================================
# 3. Forward Pass with Verification
# ==============================================================================

println("\n3. Running forward pass with verification...")

# Create test input
x = randn(Float32, 8, 20)  # Batch of 8 samples, 20 features
println("   Input shape: $(size(x))")

# Forward pass
y = model2(x)
println("   Output shape: $(size(y.data))")

# Verify output properties
println("\n   Verifying output properties:")
println("   - Probabilities sum to 1: $(all(isapprox.(sum(y.data, dims=2), 1.0, atol=1e-5)) ? "✓" : "✗")")
println("   - All non-negative: $(all(y.data .>= 0) ? "✓" : "✗")")
println("   - No NaN values: $(!any(isnan, y.data) ? "✓" : "✗")")

# ==============================================================================
# 4. Training Example
# ==============================================================================

println("\n4. Training the model...")

# Generate synthetic data
X, y_labels = make_moons(500, noise=0.2)
y_onehot = one_hot(y_labels, 2)

# Split data
train_data, test_data = train_test_split((X, y_onehot), test_ratio=0.2)
train_loader = DataLoader(train_data, batch_size=32, shuffle=true)

# Create fresh model and optimizer
model = Sequential(
    Dense(2, 16, relu),
    Dense(16, 8, relu),
    Dense(8, 2),
    Softmax()
)
optimizer = Adam(lr=0.01f0)

# Training loop (simplified)
println("\n   Training for 10 epochs...")
for epoch in 1:10
    total_loss = 0.0
    n_batches = 0

    for (batch_x, batch_y) in train_loader
        pred = model(batch_x).data
        loss = crossentropy(pred, batch_y)
        total_loss += loss
        n_batches += 1
    end

    if epoch % 2 == 0
        println("   Epoch $epoch: Loss = $(round(total_loss / n_batches, digits=4))")
    end
end

# ==============================================================================
# 5. Formal Verification
# ==============================================================================

println("\n5. Running formal verification...")

result = verify(model,
    properties=[
        ValidProbabilities(),
        FiniteOutput(),
        NoNaN()
    ],
    data=collect(DataLoader(test_data, batch_size=32))
)

println("\n$result")

# ==============================================================================
# 6. Generate Certificate
# ==============================================================================

println("\n6. Generating verification certificate...")

if result.passed
    cert = generate_certificate(model, result, model_name="BinaryClassifier")
    println("\n$cert")
end

# ==============================================================================
# Summary
# ==============================================================================

println("\n" * "═" ^ 60)
println("  Summary")
println("═" ^ 60)
println("
  Axiom.jl provides:

  1. @axiom DSL for declarative model definition
  2. Compile-time shape verification
  3. @ensure for runtime guarantees
  4. @prove for formal proofs (experimental)
  5. Verification certificates for deployment
  6. Rust backend for production performance

  Next steps:
  - Install Rust backend for accelerated production kernels
  - Use `from_pytorch(...)` and `to_onnx(...)` for interop workflows
  - Deploy verified models with confidence
")
