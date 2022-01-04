module ego_model
export EgoModel

using Knet

push!(LOAD_PATH, pwd());
using models: Dense

struct EgoModel
    dt::Float32
    front_wb::Param
    rear_wb::Param
    steer_gain::Param
    brake_accel::Param
    throt_accel::Dense

    function EgoModel(dt=1/4)
        _atype = Knet.CUDA.functional() ? KnetArray{Float32} : Array{Float32}
        front_wb = Param(_atype(ones(1)))
        rear_wb = Param(_atype(ones(1)))

        steer_gain = Param(_atype(ones(1)))
        brake_accel = Param(_atype(zeros(1)))

        throt_accel = Dense(1,1)
        new(dt,front_wb,rear_wb,steer_gain,brake_accel, throt_accel)
    end
end

(m::EgoModel)(locs, yaws, spds, acts) =
begin
    steer = selectdim(acts, length(size(acts)), 1:1)
    throt = selectdim(acts, length(size(acts)), 2:2)
    brake = convert(KnetArray{UInt8}, selectdim(acts, length(size(acts)), 3:3))
    
    # accel = #TODO: do acceleration calculation
    wheel = m.steer_gain .* steer

    beta = atan.(m.rear_wb/(m.front_wb + m.rear_wb) .* wheel)
    next_locs = locs + spds .* cat(cos.(yaws+beta), sin.(yaws+beta);dims=length(size(beta)))
    next_yaws = yaws + spds ./ m.rear_wb .* sin.(beta) * m.dt
    next_spds = spds + dt .* accel
    return next_locs, next_yaws, relu(next_spds)
end
end

