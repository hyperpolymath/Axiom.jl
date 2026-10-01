# SPDX-License-Identifier: MPL-2.0
# Probe Zig activation overflow/cancellation against Float64 reference formulas.
include("probe_common.jl")

function activation(symbol, x)
  y = zeros(Float32, size(x))
  ccall(Libdl.dlsym(lib, symbol), Cvoid,
        (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t), x, y, length(x))
  return y
end

gelu_reference(x) = Float32(0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))))
xs = Float32[1e-8, 1e-5, 1e-3, 5, 9, 10, 10.5, 11, 12, 20, 44, 44.3,
              44.5, 45, 50, 100, -50, -100, Inf, -Inf]
zt, zg, zs = activation(:axiom_tanh, xs), activation(:axiom_gelu, xs), activation(:axiom_sigmoid, xs)
rt = Float32.(tanh.(Float64.(xs)))
rg = gelu_reference.(Float64.(xs))
rs = Float32.(1 ./ (1 .+ exp.(-Float64.(xs))))
println("         x |     zig tanh     ref tanh |     zig gelu     ref gelu |  zig sigmoid          ref")
for i in eachindex(xs)
  flag = ""
  !isfinite(zt[i]) && isfinite(rt[i]) && (flag *= " TANH-NaN")
  !isfinite(zg[i]) && isfinite(rg[i]) && (flag *= " GELU-NaN")
  if isfinite(zt[i]) && rt[i] != 0 && abs(zt[i] - rt[i]) / abs(rt[i]) > 1e-4
    flag *= @sprintf(" tanh-relerr=%.1e", abs(zt[i] - rt[i]) / abs(rt[i]))
  end
  @printf("%10.4g | %12.6g %12.6g | %12.6g %12.6g | %12.6g %12.6g%s\n",
          xs[i], zt[i], rt[i], zg[i], rg[i], zs[i], rs[i], flag)
end
xt = copy(xs)
ccall(Libdl.dlsym(lib, :axiom_tanh_inplace), Cvoid, (Ptr{Cfloat}, Csize_t), xt, length(xt))
println("\naxiom_tanh_inplace(50) = ", xt[findfirst(==(50), xs)],
        " axiom_tanh_inplace(11) = ", xt[findfirst(==(11), xs)])
rng = MersenneTwister(0)
for sigma in (1, 3, 5, 8)
  pre = Float32.(randn(rng, 1_000_000) .* sigma)
  g = activation(:axiom_gelu, pre)
  @printf("gelu on N(0,%d²) x1e6: NaN count = %d, max|x|=%.1f\n",
          sigma, count(isnan, g), maximum(abs, pre))
end
