# SPDX-License-Identifier: MPL-2.0
# Probe layernorm row correctness and an output canary for batch=5.
include("probe_common.jl")

for (batch, hidden) in ((5, 2048), (5, 4096), (5, 65536))
  x = Float32.(randn(MersenneTwister(0), hidden, batch))
  canary = 8 * hidden
  ybuf = fill(12345f0, batch * hidden + canary)
  gamma, beta = ones(Float32, hidden), zeros(Float32, hidden)
  ccall(Libdl.dlsym(lib, :axiom_layernorm), Cvoid,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Cfloat),
        x, ybuf, gamma, beta, batch, hidden, 1f-5)
  y = reshape(view(ybuf, 1:batch * hidden), hidden, batch)
  reference = layernorm_reference(x)
  bad_rows = count(any(.!close_element.(y, reference; atol=1e-3); dims=1))
  overwritten = count(!=(12345f0), view(ybuf, batch * hidden + 1:length(ybuf)))
  println("B=$batch H=$hidden: bad_rows=$bad_rows canary_overwritten=$overwritten/$canary")
end
