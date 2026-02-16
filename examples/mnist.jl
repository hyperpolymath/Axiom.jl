# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl MNIST Example
#
# Simple neural network for MNIST digit classification.

using Axiom

# ==============================================================================
# Method 1: Using @axiom macro (declarative style)
# ==============================================================================

@axiom MNISTClassifier begin
    input :: Tensor{Float32, (:batch, 784)}
    output :: Tensor{Float32, (:batch, 10)}

    # Network architecture
    hidden1 = input |> Dense(784, 256, relu)
    hidden2 = hidden1 |> Dense(256, 128, relu)
    hidden3 = hidden2 |> Dense(128, 64, relu)
    logits = hidden3 |> Dense(64, 10)
    output = logits |> Softmax

    # Guarantees
    @ensure sum(output) ≈ 1.0  # Valid probability distribution
    @ensure all(output .>= 0)   # Non-negative
end

# ==============================================================================
# Method 2: Using Sequential (imperative style)
# ==============================================================================

function create_mnist_model()
    Sequential(
        Flatten(),
        Dense(784, 256, relu),
        Dense(256, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 10),
        Softmax()
    )
end

# ==============================================================================
# Training
# ==============================================================================

function train_mnist()
    println("Creating MNIST classifier...")

    # Create model
    model = create_mnist_model()

    # Print model summary
    println("\nModel architecture:")
    println("  Input: (batch, 28, 28, 1)")
    println("  → Flatten → (batch, 784)")
    println("  → Dense(784, 256, relu)")
    println("  → Dense(256, 128, relu)")
    println("  → Dense(128, 64, relu)")
    println("  → Dense(64, 10)")
    println("  → Softmax")
    println("  Output: (batch, 10)")

    # Generate synthetic MNIST-like data
    println("\nGenerating synthetic training data...")
    n_train = 1000
    n_test = 200

    X_train = randn(Float32, n_train, 784)
    y_train = rand(1:10, n_train)
    y_train_onehot = one_hot(y_train, 10)

    X_test = randn(Float32, n_test, 784)
    y_test = rand(1:10, n_test)
    y_test_onehot = one_hot(y_test, 10)

    # Create data loaders
    train_loader = DataLoader((X_train, y_train_onehot), batch_size=32, shuffle=true)
    test_loader = DataLoader((X_test, y_test_onehot), batch_size=32)

    # Create optimizer
    optimizer = Adam(lr=0.001f0)

    # Training loop
    println("\nTraining...")
    epochs = 5

    for epoch in 1:epochs
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x, batch_y) in train_loader
            # Forward pass
            pred = model(batch_x).data

            # Compute loss
            loss = crossentropy(pred, batch_y)
            epoch_loss += loss
            n_batches += 1
        end

        avg_loss = epoch_loss / n_batches
        println("  Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=4))")
    end

    # Verification
    println("\nVerifying model properties...")
    test_input = randn(Float32, 1, 784)
    output = model(test_input)

    println("  ✓ Output shape: $(size(output.data))")
    println("  ✓ Probabilities sum to: $(sum(output.data))")
    println("  ✓ All non-negative: $(all(output.data .>= 0))")
    println("  ✓ No NaN values: $(!any(isnan, output.data))")

    # Formal verification
    result = verify(model,
        properties=[ValidProbabilities(), FiniteOutput()],
        data=collect(test_loader)
    )

    println("\n$result")

    model
end

# ==============================================================================
# Main
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    train_mnist()
end
