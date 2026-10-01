# SPDX-License-Identifier: MPL-2.0
# Probe the fixed 4096-feature batchnorm scratch, isolating stack corruption.
include("probe_common.jl")

function child(features)
  batch = 2
  x = Float32.(randn(MersenneTwister(0), features, batch))
  y = zeros(Float32, size(x))
  gamma, beta = ones(Float32, features), zeros(Float32, features)
  mean, variance = zeros(Float32, features), ones(Float32, features)
  ccall(Libdl.dlsym(lib, :axiom_batchnorm), Cvoid,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
          Csize_t, Csize_t, Cfloat),
        x, y, gamma, beta, mean, variance, length(x), features, 1f-5)
  reference = (x .- mean) ./ sqrt.(variance .+ 1f-5)
  println(count(!, close_element.(y, reference; atol=1e-4)), " of ", length(x))
end

if length(ARGS) > 1 && ARGS[2] == "--child"
  child(parse(Int, ARGS[3]))
else
  for features in (4096, 4097, 4100, 8192, 65536, 1_000_000)
    @printf("num_features=%8d: %s\n", features, run_case(features))
  end
end
