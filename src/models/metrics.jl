module metrics
export l1_loss

using Statistics: mean

function l1_loss(x::AbstractArray, y::AbstractArray)
    return mean(x-y)
end