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
    
end

end

