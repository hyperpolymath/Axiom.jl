# SPDX-License-Identifier: MPL-2.0
# Probe softmax/layernorm chunking, isolating potential ReleaseFast crashes.
include("probe_common.jl")

function child(batch, hidden, operation)
  x = Float32.(randn(MersenneTwister(0), hidden, batch))
  y = fill(7f0, size(x))
  if operation == "softmax"
    ccall(Libdl.dlsym(lib, :axiom_softmax), Cvoid,
          (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t), x, y, batch, hidden)
    reference = exp.(x .- maximum(x; dims=1))
    reference ./= sum(reference; dims=1)
  elseif operation == "layernorm"
    gamma, beta = ones(Float32, hidden), zeros(Float32, hidden)
    ccall(Libdl.dlsym(lib, :axiom_layernorm), Cvoid,
          (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Cfloat),
          x, y, gamma, beta, batch, hidden, 1f-5)
    reference = layernorm_reference(x)
  else
    error("Unknown operation: $operation")
  end
  println("mismatches=", count(!, close_element.(y, reference; atol=1e-4)))
end

if length(ARGS) > 1 && ARGS[2] == "--child"
  child(parse(Int, ARGS[3]), parse(Int, ARGS[4]), ARGS[5])
else
  for operation in ("softmax", "layernorm"), batch in (4, 5, 6, 7, 8, 9, 13, 17)
    hidden = 2048
    @printf("%9s B=%2d H=%d: %s\n", operation, batch, hidden, run_case(batch, hidden, operation))
  end
end
