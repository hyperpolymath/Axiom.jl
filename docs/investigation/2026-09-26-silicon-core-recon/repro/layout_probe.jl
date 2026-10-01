# SPDX-License-Identifier: MPL-2.0
# Pass column-major NHWC buffers directly to row-major Zig pooling kernels,
# as the investigated Julia wrappers did. Preserve this layout mismatch.
include("probe_common.jl")

function reference_maxpool(x, kernel, stride)
  batch, height, width, channels = size(x)
  out_h, out_w = (height - kernel) ÷ stride + 1, (width - kernel) ÷ stride + 1
  y = Array{Float32}(undef, batch, out_h, out_w, channels)
  for n in 1:batch, c in 1:channels, i in 1:out_h, j in 1:out_w
    rows = (i - 1) * stride + 1:(i - 1) * stride + kernel
    cols = (j - 1) * stride + 1:(j - 1) * stride + kernel
    y[n, i, j, c] = maximum(view(x, n, rows, cols, c))
  end
  return y
end

function julia_style_maxpool(x, kernel, stride)
  batch, height, width, channels = size(x)
  out_h, out_w = (height - kernel) ÷ stride + 1, (width - kernel) ÷ stride + 1
  y = zeros(Float32, batch, out_h, out_w, channels)
  ccall(Libdl.dlsym(lib, :axiom_maxpool2d), Cvoid,
        (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Csize_t, Csize_t,
          Csize_t, Csize_t, Csize_t, Csize_t),
        x, y, batch, height, width, channels, kernel, kernel, stride, stride)
  return y
end

function julia_style_gap(x)
  batch, height, width, channels = size(x)
  y = zeros(Float32, batch, channels)
  ccall(Libdl.dlsym(lib, :axiom_global_avgpool2d), Cvoid,
        (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Csize_t, Csize_t),
        x, y, batch, height, width, channels)
  return y
end

rng = MersenneTwister(0)
println("case                      | max|zig-ref| | mismatched elements")
for (batch, height, width, channels, kernel, stride) in
    ((1, 4, 4, 1, 2, 2), (1, 4, 4, 2, 2, 2), (2, 4, 4, 1, 2, 2),
      (2, 6, 6, 3, 2, 2), (1, 5, 3, 1, 2, 1))
  x = Float32.(randn(rng, batch, height, width, channels))
  reference = reference_maxpool(x, kernel, stride)
  got = julia_style_maxpool(x, kernel, stride)
  bad = count(!, close_element.(reference, got))
  @printf("maxpool N=%d H=%d W=%d C=%d k=%d s=%d | %10.4f | %d/%d\n",
          batch, height, width, channels, kernel, stride, maximum(abs.(reference .- got)), bad, length(reference))
end
for (batch, height, width, channels) in ((1, 3, 3, 1), (1, 3, 3, 2), (2, 3, 3, 1), (2, 3, 3, 4))
  x = Float32.(randn(rng, batch, height, width, channels))
  reference = dropdims(sum(x; dims=(2, 3)); dims=(2, 3)) ./ (height * width)
  got = julia_style_gap(x)
  bad = count(!, close_element.(reference, got))
  @printf("gap     N=%d H=%d W=%d C=%d         | %10.4f | %d/%d\n",
          batch, height, width, channels, maximum(abs.(reference .- got)), bad, length(reference))
end
println("\nEdge-case contracts (single-channel windows so layout is not a factor):")
x = fill(-Inf32, 1, 2, 2, 1)
println("  all -Inf window -> zig=", only(julia_style_maxpool(x, 2, 2)), " julia(maximum)=-Inf")
x = reshape(Float32[1, NaN, 0.5, 0.25], 1, 2, 2, 1)
println("  window with NaN -> zig=", only(julia_style_maxpool(x, 2, 2)), " julia(maximum)=NaN")
