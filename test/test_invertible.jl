# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Invertible / Reversible Computing Tests

@testset "Invertible Layers" begin

    @testset "CouplingLayer" begin
        dim = 10
        hidden = 32
        batch = 8

        layer = CouplingLayer(dim, hidden)
        x = randn(Float32, batch, dim)

        # Forward produces correct shape
        y = Axiom.forward(layer, x)
        @test size(y) == (batch, dim)
        @test !any(isnan, y)

        # Roundtrip: inverse(forward(x)) ≈ x
        x_rec = Axiom.inverse(layer, y)
        @test isapprox(x_rec, x, atol=1e-5)

        # Forward-and-log-det
        y2, ld = Axiom.forward_and_log_det(layer, x)
        @test isapprox(y2, y, atol=1e-6)
        @test length(ld) == batch
        @test !any(isnan, ld)

        # Log-det standalone matches fused
        ld2 = Axiom.log_abs_det_jacobian(layer, x)
        @test isapprox(ld, ld2, atol=1e-6)

        # Parameters returns NamedTuple
        p = Axiom.parameters(layer)
        @test haskey(p, :scale_net_w1)
        @test haskey(p, :translate_net_w2)

        # Tensor interface
        xt = Tensor(x)
        yt = Axiom.forward(layer, xt)
        @test yt isa Tensor
        @test isapprox(yt.data, y, atol=1e-6)

        xt_rec = Axiom.inverse(layer, yt)
        @test isapprox(xt_rec.data, x, atol=1e-5)

        # Mask parity variant
        layer_p = CouplingLayer(dim, hidden, mask_parity=true)
        y_p = Axiom.forward(layer_p, x)
        x_p_rec = Axiom.inverse(layer_p, y_p)
        @test isapprox(x_p_rec, x, atol=1e-5)
    end

    @testset "ActNorm" begin
        dim = 8
        batch = 16

        layer = ActNorm(dim)
        @test !layer.initialized

        x = randn(Float32, batch, dim) .* 3.0f0 .+ 5.0f0
        y = Axiom.forward(layer, x)
        @test layer.initialized
        @test size(y) == (batch, dim)

        # After data-dependent init, output should be roughly normalized
        @test isapprox(mean(y), 0.0f0, atol=0.5)

        # Roundtrip
        x_rec = Axiom.inverse(layer, y)
        @test isapprox(x_rec, x, atol=1e-5)

        # Log-det
        ld = Axiom.log_abs_det_jacobian(layer, x)
        @test length(ld) == batch
        # All elements should be the same (per-sample log-det is constant for ActNorm)
        @test all(isapprox.(ld, ld[1], atol=1e-6))
        @test !any(isnan, ld)

        # Forward-and-log-det
        y2, ld2 = Axiom.forward_and_log_det(layer, x)
        @test isapprox(y2, y, atol=1e-6)
        @test isapprox(ld2, ld, atol=1e-6)

        # Tensor interface
        xt = Tensor(x)
        yt = Axiom.forward(layer, xt)
        @test yt isa Tensor
        xt_rec = Axiom.inverse(layer, yt)
        @test isapprox(xt_rec.data, x, atol=1e-5)
    end

    @testset "Invertible1x1Conv" begin
        dim = 6
        batch = 4

        layer = Invertible1x1Conv(dim)
        x = randn(Float32, batch, dim)

        # Forward
        y = Axiom.forward(layer, x)
        @test size(y) == (batch, dim)
        @test !any(isnan, y)

        # Roundtrip
        x_rec = Axiom.inverse(layer, y)
        @test isapprox(x_rec, x, atol=1e-4)

        # Log-det (should be constant across batch)
        ld = Axiom.log_abs_det_jacobian(layer, x)
        @test length(ld) == batch
        @test all(isapprox.(ld, ld[1], atol=1e-6))
        @test isfinite(ld[1])

        # Forward-and-log-det
        y2, ld2 = Axiom.forward_and_log_det(layer, x)
        @test isapprox(y2, y, atol=1e-6)
        @test isapprox(ld2, ld, atol=1e-6)

        # Parameters
        p = Axiom.parameters(layer)
        @test haskey(p, :L_lower)
        @test haskey(p, :U_upper)
        @test haskey(p, :log_s)

        # Tensor interface
        xt = Tensor(x)
        yt = Axiom.forward(layer, xt)
        @test yt isa Tensor
        xt_rec = Axiom.inverse(layer, yt)
        @test isapprox(xt_rec.data, x, atol=1e-4)
    end

    @testset "RevBlock" begin
        half_dim = 5
        full_dim = half_dim * 2
        batch = 8

        F_net = Dense(half_dim, half_dim, relu)
        G_net = Dense(half_dim, half_dim, relu)
        layer = RevBlock(F_net, G_net)

        x = randn(Float32, batch, full_dim)

        # Forward
        y = Axiom.forward(layer, x)
        @test size(y) == (batch, full_dim)
        @test !any(isnan, y)

        # Roundtrip
        x_rec = Axiom.inverse(layer, y)
        @test isapprox(x_rec, x, atol=1e-4)

        # Volume-preserving: log-det = 0
        ld = Axiom.log_abs_det_jacobian(layer, x)
        @test all(isapprox.(ld, 0.0f0, atol=1e-6))

        # Forward-and-log-det
        y2, ld2 = Axiom.forward_and_log_det(layer, x)
        @test isapprox(y2, y, atol=1e-6)
        @test all(isapprox.(ld2, 0.0f0, atol=1e-6))

        # Parameters contain F and G sub-parameters
        p = Axiom.parameters(layer)
        @test haskey(p, :F)
        @test haskey(p, :G)

        # Tensor interface
        xt = Tensor(x)
        yt = Axiom.forward(layer, xt)
        @test yt isa Tensor
        xt_rec = Axiom.inverse(layer, yt)
        @test isapprox(xt_rec.data, x, atol=1e-4)
    end

    @testset "InvertibleSequential" begin
        dim = 8
        batch = 4

        seq = InvertibleSequential(
            ActNorm(dim),
            CouplingLayer(dim, 16),
            CouplingLayer(dim, 16, mask_parity=true),
        )

        x = randn(Float32, batch, dim)

        # Forward
        y = Axiom.forward(seq, x)
        @test size(y) == (batch, dim)

        # Roundtrip
        x_rec = Axiom.inverse(seq, y)
        @test isapprox(x_rec, x, atol=1e-4)

        # Forward-and-log-det
        y2, ld = Axiom.forward_and_log_det(seq, x)
        @test isapprox(y2, y, atol=1e-5)
        @test length(ld) == batch
        @test !any(isnan, ld)
        @test all(isfinite, ld)

        # Log-det standalone
        ld2 = Axiom.log_abs_det_jacobian(seq, x)
        @test isapprox(ld, ld2, atol=1e-4)

        # Parameters
        p = Axiom.parameters(seq)
        @test length(p) == 3

        # Tensor interface
        xt = Tensor(x)
        yt = Axiom.forward(seq, xt)
        @test yt isa Tensor
        xt_rec = Axiom.inverse(seq, yt)
        @test isapprox(xt_rec.data, x, atol=1e-4)
    end

    @testset "NormalizingFlow" begin
        dim = 6
        batch = 16

        transform = InvertibleSequential(
            ActNorm(dim),
            CouplingLayer(dim, 16),
            CouplingLayer(dim, 16, mask_parity=true),
        )
        nf = NormalizingFlow(transform, base_dim=dim)

        x = randn(Float32, batch, dim)

        # Log-prob returns one value per sample
        lp = Axiom.flow_log_prob(nf, x)
        @test length(lp) == batch
        @test all(isfinite, lp)
        @test !any(isnan, lp)

        # Tensor interface for log-prob
        xt = Tensor(x)
        lp_t = Axiom.flow_log_prob(nf, xt)
        @test isapprox(lp_t, lp, atol=1e-5)

        # Sampling
        samples = Axiom.flow_sample(nf, 32)
        @test size(samples) == (32, dim)
        @test all(isfinite, samples)

        # NLL loss is a scalar
        nll = Axiom.flow_nll_loss(nf, x)
        @test isfinite(nll)
        @test nll isa AbstractFloat

        # NLL with Tensor
        nll_t = Axiom.flow_nll_loss(nf, xt)
        @test isapprox(nll_t, nll, atol=1e-5)
    end

    @testset "Gradient Flow" begin
        dim = 6
        batch = 4

        # CouplingLayer gradient
        cl = CouplingLayer(dim, 16)
        x = randn(Float32, batch, dim)
        g = Axiom.gradient(x -> sum(Axiom.forward(cl, x)), x)
        @test g[1] !== nothing
        @test size(g[1]) == size(x)
        @test all(isfinite, g[1])

        # ActNorm gradient
        an = ActNorm(dim)
        Axiom.forward(an, x)  # initialize
        g_an = Axiom.gradient(x -> sum(Axiom.forward(an, x)), x)
        @test g_an[1] !== nothing
        @test all(isfinite, g_an[1])

        # NormalizingFlow NLL gradient (the main use case)
        transform = InvertibleSequential(
            ActNorm(dim),
            CouplingLayer(dim, 16),
        )
        nf = NormalizingFlow(transform, base_dim=dim)
        # Initialize ActNorm
        Axiom.flow_log_prob(nf, x)
        g_nf = Axiom.gradient(x -> Axiom.flow_nll_loss(nf, x), x)
        @test g_nf[1] !== nothing
        @test size(g_nf[1]) == size(x)
        @test all(isfinite, g_nf[1])
    end

    @testset "Pipeline Compatibility" begin
        # InvertibleLayer works inside Sequential
        dim = 8
        batch = 4

        model = Sequential(
            Dense(20, dim, Axiom.relu),
            CouplingLayer(dim, 16),
            Dense(dim, 3),
            Softmax()
        )

        x = Tensor(randn(Float32, batch, 20))
        y = model(x)
        @test size(y) == (batch, 3)
        @test all(isapprox.(sum(y.data, dims=2), 1.0f0, atol=1e-5))
    end

    @testset "Numerical Log-Det Verification" begin
        # Verify analytic log-det against numerical full Jacobian determinant
        # (small dimensions only — O(D³) for full det)
        dim = 4
        batch = 2

        cl = CouplingLayer(dim, 8)
        x = randn(Float32, batch, dim)

        # Analytic log-det
        ld_analytic = Axiom.log_abs_det_jacobian(cl, x)

        # Numerical: compute full Jacobian per sample and take log |det|
        for i in 1:batch
            xi = x[i:i, :]
            J = zeros(Float32, dim, dim)
            for j in 1:dim
                e_j = zeros(Float32, 1, dim)
                e_j[1, j] = 1.0f0
                # Finite difference Jacobian column
                eps = Float32(1e-4)
                y_plus = Axiom.forward(cl, xi .+ eps .* e_j)
                y_minus = Axiom.forward(cl, xi .- eps .* e_j)
                J[:, j] = vec((y_plus .- y_minus) ./ (2 * eps))
            end
            ld_numerical = log(abs(det(J)))
            @test isapprox(ld_analytic[i], ld_numerical, atol=0.05)
        end
    end

end
