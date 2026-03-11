# SPDX-License-Identifier: PMPL-1.0-or-later
#
# Axiom.jl Custom Zygote Adjoints for Invertible Layers
# ======================================================
#
# Memory-efficient backpropagation for reversible computing layers.
#
# Key insight: for invertible layers, we can recompute activations from outputs
# via the inverse transform during the backward pass, avoiding the need to store
# intermediate activations. This trades compute for memory — critical for deep
# normalizing flows and reversible networks.
#
# Custom rules defined here:
# - RevBlock: Recomputes (x1, x2) from (y1, y2) via inverse during backward
# - InvertibleSequential: Recomputes per-layer inputs via inverse chain

using Zygote

# ---------------------------------------------------------------------------
# RevBlock: Memory-efficient backward via inverse reconstruction
# ---------------------------------------------------------------------------
#
# Standard backprop through a RevBlock stores x1, x2, F(x2), G(y1) — 4 tensors.
# With the custom adjoint we store only (y1, y2) and recompute x1, x2 via inverse.

Zygote.@adjoint function _forward_data(layer::RevBlock, x::AbstractMatrix)
    # Forward pass: just compute normally
    y = _forward_data(layer, x)

    function revblock_pullback(Δy)
        # Reconstruct inputs from outputs via inverse (no stored intermediates)
        y1, y2 = _revblock_split(y)
        Δy1, Δy2 = _revblock_split(Δy)

        # Reconstruct x2 = y2 - G(y1)
        G_y1 = _layer_forward_data(layer.G, y1)
        x2 = y2 .- G_y1

        # Reconstruct x1 = y1 - F(x2)
        F_x2 = _layer_forward_data(layer.F, x2)
        x1 = y1 .- F_x2

        # Backprop through G: Δ_G wrt y1 and G's params
        # dy2/dG_y1 = 1, so Δ_G_output = Δy2
        # dy2/dy1 via G: need Zygote gradient of G wrt y1
        _, G_back = Zygote.pullback(z -> _layer_forward_data(layer.G, z), y1)
        Δ_G_input = G_back(Δy2)[1]

        # Backprop through F: Δ_F wrt x2 and F's params
        _, F_back = Zygote.pullback(z -> _layer_forward_data(layer.F, z), x2)
        Δ_F_input = F_back(Δy1)[1]

        # Aggregate gradients for x1 and x2
        Δx1 = Δy1  # dy1/dx1 = I
        Δx2 = Δy2 .+ Δ_F_input  # from both y2 and y1 paths

        # Add contribution of G's dependence on y1 which depends on x1
        Δx1 = Δx1 .+ Δ_G_input

        Δx = _revblock_merge(Δx1, Δx2)

        return (nothing, Δx)
    end

    return y, revblock_pullback
end

# ---------------------------------------------------------------------------
# InvertibleSequential: Layer-by-layer reverse reconstruction
# ---------------------------------------------------------------------------
#
# For a chain of N invertible layers, standard backprop stores N intermediate
# activations. With the custom adjoint, we store only the final output and
# reconstruct each intermediate via inverse during the backward pass.

Zygote.@adjoint function forward_and_log_det(seq::InvertibleSequential, x::AbstractMatrix)
    # Forward pass: store final output and total log-det
    y, total_ld = forward_and_log_det(seq, x)

    function seq_pullback(Δ)
        Δy, Δld = Δ
        layers = collect(seq.layers)
        n = length(layers)

        # Reconstruct intermediate values by inverting from output
        # We need the input to each layer for gradient computation
        # Reconstruct backwards: z_n = y, z_{k} = inverse(layer_{k+1}, z_{k+1})
        intermediates = Vector{AbstractMatrix}(undef, n + 1)
        intermediates[n + 1] = y isa Tensor ? y.data : y

        for k in n:-1:1
            intermediates[k] = inverse(layers[k], intermediates[k + 1])
        end

        # Now backprop through each layer in reverse
        Δ_current = Δy isa Tensor ? Δy.data : Δy
        for k in n:-1:1
            layer_k = layers[k]
            x_k = intermediates[k]

            # Get gradient through this layer's forward_and_log_det
            _, layer_back = Zygote.pullback(
                z -> begin
                    fwd, ld = forward_and_log_det(layer_k, z)
                    fwd_data = fwd isa Tensor ? fwd.data : fwd
                    (fwd_data, ld)
                end,
                x_k
            )

            # Combine output gradient with log-det gradient
            Δ_layer = layer_back((Δ_current, Δld))
            Δ_current = Δ_layer[1]
        end

        return (nothing, Δ_current)
    end

    return (y, total_ld), seq_pullback
end
