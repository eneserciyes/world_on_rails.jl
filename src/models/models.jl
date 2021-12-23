module models
export Conv, Dense, Chain

using Knet, Statistics, Random

# Defining the convolutional layer:
struct Conv
    w
    b
    f
    p
end
(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x, c.p)) .+ c.b))
Conv(w1::Int, w2::Int, cx::Int, cy::Int, f = relu; pdrop = 0) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f, pdrop)

# Defining dense layer
struct Dense
    w
    b
    f
    p
end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int, o::Int, f = relu; pdrop = 0) = Dense(param(o, i), param0(o), f, pdrop)

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers
    x = l(x)
end;
x)
(c::Chain)(x, y) = nll(c(x), y)

end