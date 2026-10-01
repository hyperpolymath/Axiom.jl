# SPDX-License-Identifier: MPL-2.0
# Shared standard-library support for the standalone investigation probes.
using Libdl, Random, Printf

isempty(ARGS) && error("Usage: julia <probe>.jl /path/to/libaxiom_zig.so")
const lib = Libdl.dlopen(abspath(ARGS[1]))

# Match NumPy's elementwise isclose defaults (not Julia's norm-based isapprox).
close_element(a, b; atol=1e-8, rtol=1e-5) =
  a == b || (isfinite(a) && isfinite(b) && abs(a - b) <= atol + rtol * abs(b))

# Columns are independent rows of the C row-major (batch, hidden) buffer.
function layernorm_reference(x)
  centered = x .- sum(x; dims=1) ./ size(x, 1)
  variance = sum(abs2, centered; dims=1) ./ size(x, 1)
  return centered ./ sqrt.(variance .+ 1f-5)
end

function run_case(args...)
  stdout_buffer, stderr_buffer = IOBuffer(), IOBuffer()
  command = `$(Base.julia_cmd()) --startup-file=no $(abspath(PROGRAM_FILE)) $(ARGS[1]) --child $args`
  process = run(pipeline(ignorestatus(command); stdout=stdout_buffer, stderr=stderr_buffer))
  output = strip(String(take!(stdout_buffer)))
  errors = strip(String(take!(stderr_buffer)))
  if success(process)
    return "exit=0 " * output
  end
  detail = isempty(errors) ? "signal" : first(last(split(errors, '\n')), 80)
  return "CRASH exit=$(process.exitcode) signal=$(process.termsignal) ($detail)"
end
