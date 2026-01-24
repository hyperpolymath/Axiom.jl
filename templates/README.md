# Axiom.jl Model Zoo Templates

Verification-ready model templates with formal properties and metadata.

## Quick Start

### Computer Vision

```julia
using Axiom

# ResNet50 for image classification
model, metadata = resnet50_verified(pretrained=true)

# Verify properties
@prove ∀x ∈ ImageNet. is_finite(model(x))

# Inference
predictions = model(load_image("cat.jpg"))
```

### Natural Language Processing

```julia
using Axiom

# Transformer encoder
model, metadata = transformer_encoder_verified(
    vocab_size=50000,
    d_model=768,
    n_heads=12
)

# Verify attention properties
@prove ∀x. sum(attention_weights(model, x), dims=2) ≈ 1.0

# Inference
embeddings = model(tokenize("Hello world"))
```

## Available Templates

### Computer Vision

| Model | Task | Parameters | Verification Claims |
|-------|------|------------|---------------------|
| ResNet50 | Image Classification | 25.6M | Lipschitz, finite outputs, probability bounds |
| MobileNetV2 | Mobile CV | 3.5M | Efficient inference, quantization-ready |
| EfficientNet-B0 | Compound Scaling | 5.3M | Accuracy-efficiency tradeoffs |
| Vision Transformer (ViT) | Image Classification | 86M | Attention normalization, patch embedding |

### Natural Language Processing

| Model | Task | Parameters | Verification Claims |
|-------|------|------------|---------------------|
| Transformer Encoder | Sequence Modeling | Custom | Attention normalized, stable embeddings |
| BERT-Base | Masked LM | 110M | Token prediction bounds, attention sparsity |
| GPT-2 Small | Autoregressive LM | 117M | Causality preserved, generation stability |
| T5-Small | Seq2Seq | 60M | Encoder-decoder alignment |

## Model Structure

Each template provides:

1. **Verified Architecture** - Built with `@axiom` macro for shape checking
2. **Metadata** - Complete `ModelMetadata` with provenance
3. **Verification Claims** - Formal properties verified with `@prove`
4. **Pretrained Weights** - Optional pretrained checkpoints
5. **Documentation** - Usage examples and property specifications

## Creating Custom Templates

### Basic Template Structure

```julia
function my_model_verified(; pretrained=false, num_classes=10)
    # Define architecture
    model = @axiom begin
        # ... layers ...
    end

    # Load weights if pretrained
    if pretrained
        load_weights!(model, "path/to/weights.jld2")
    end

    # Create metadata
    metadata = create_metadata(
        model,
        name="MyModel",
        architecture="Custom",
        task="classification",
        # ... other fields ...
    )

    # Add verification claims
    verify_and_claim!(
        metadata,
        "Property name",
        "Formal specification"
    )

    # Save metadata
    save_metadata(metadata, "my_model_metadata.json")

    return model, metadata
end
```

### Verification Best Practices

1. **Start with basic properties**:
   - No NaN/Inf propagation
   - Output bounds (for probabilities, etc.)
   - Input-output relationship preservation

2. **Add domain-specific properties**:
   - CV: Lipschitz continuity, spatial invariance
   - NLP: Attention normalization, causality

3. **Test verification overhead**:
   - Profile verification time
   - Consider caching verified properties
   - Use sampling for large input spaces

4. **Document limitations**:
   - Approximations made
   - Assumptions about input distribution
   - Verification scope (training vs inference)

## Extending Templates

### Adding New Architecture

1. Create file in `templates/{cv,nlp}/my_architecture.jl`
2. Implement `my_architecture_verified()` function
3. Add verification claims appropriate to architecture
4. Create metadata with complete provenance
5. Add tests in `test/templates/`
6. Document in this README

### Adding Pretrained Weights

Weights should be:
- Stored in standard format (JLD2, BSON, or HuggingFace compatible)
- Include checksum (SHA256)
- Provide download script or registry link
- Include training configuration and metrics

### Verification Properties

Common properties to verify:

**Stability:**
- No NaN/Inf propagation
- Bounded gradients (Lipschitz)
- Numerical precision limits

**Correctness:**
- Output shape matches specification
- Probability distributions (sum to 1, non-negative)
- Attention weights normalized

**Robustness:**
- Bounded perturbation resistance
- Adversarial input handling
- Out-of-distribution detection

**Performance:**
- Inference time bounds
- Memory usage limits
- Batch processing guarantees

## Integration with HuggingFace

Import pretrained models from HuggingFace Hub:

```julia
using Axiom

# Import with automatic verification
model, metadata = from_pretrained(
    "bert-base-uncased",
    verify=true,  # Verify after import
    architecture="transformer"
)

# Verification results in metadata
for claim in metadata.verification_claims
    println("$(claim.property): $(claim.verified ? "✓" : "✗")")
end
```

## See Also

- [Model Metadata Schema](../src/model_metadata.jl)
- [HuggingFace Integration](../src/integrations/huggingface.jl)
- [Verification Guide](../docs/wiki/Verification.md)
- [Issue #15 - Verified Model Zoo](https://github.com/hyperpolymath/Axiom.jl/issues/15)
- [Issue #16 - Model Metadata](https://github.com/hyperpolymath/Axiom.jl/issues/16)
