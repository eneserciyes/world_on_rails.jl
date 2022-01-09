module metrics
export l1_loss, kl_div

using Statistics: mean
using Knet: KnetArray

function l1_loss(x::AbstractArray, y::AbstractArray)
    return mean(x-y)
end

function kl_div(input::KnetArray, target::KnetArray, reduced=true)
    # add a float for numerical stability.
    kl = target .* (log.(target .+ Float32(1e-30)) .- input)
    if reduced
        return mean(mean(reduced, 2),3)
    end
    return kl
end
end