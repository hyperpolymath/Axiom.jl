# SPDX-License-Identifier: PMPL-1.0-or-later
# Verification Module
#
# Proof certificate management and verification workflows.

module Verification

using Dates
using SHA
using JSON

# Include ProofResult from parent module
using ..Axiom: ProofResult

include("serialization.jl")

end # module Verification
