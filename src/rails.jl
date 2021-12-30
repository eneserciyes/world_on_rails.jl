module rails
export RAILS

push!(LOAD_PATH, joinpath(pwd(), "models"))
push!(LOAD_PATH, joinpath(pwd(), "data"))

using Knet: Adam
using models: set_optim!, zero_grad_model, train!
using ego_model: EgoModel
using ego_dataset: EgoDataset
using metrics: l1_loss

using YAML

struct RAILS
    config::Dict
    device::AbstractString
    ego_model::EgoModel

    function RAILS(args)
        config = nothing
        open(args.config_path, 'r') do cfg_file
            config = YAML.load(cfg_file)
        end

        ego_model = EgoModel(1 / (args.fps * (args.num_repeat + 1)))
        # set Adam optimizer to ego_model
        set_optim!(ego_model, Adam, args.lr)

        new(config, args.device, ego_model)
    end
end

function train_ego(r::RAILS, locs, rots, spds, acts)
    locs = selectdim(locs, length(size(locs)), 1:2)
    yaws = selectdim(yaws, length(size(yaws)), w:size(yaws)[end]) .* pi ./ 180

    pred_locs = []
    pred_yaws = []

    pred_loc = locs[:, 1]
    pred_yaw = yaws[:, 1]
    pred_spd = spds[:, 1]

    for t = 1:size(locs, 2)-1
        act = acts[:, t]
        pred_loc, pred_yaw, pred_spd = r.ego_model(pred_loc, pred_yaw, pred_spd, act)
        append!(pred_locs, pred_loc)
        append!(pred_yaws, pred_yaw)
    end
    loc_loss = l1_loss(pred_locs, locs[:, 2:end])
    ori_loss = l1_loss(cos.(pred_yaws), cos.(yaws[:, 2:end])) + l1_loss(sin.(pred_yaws), sin.(yaws[:, 2:end]))

    loss = @diff loc_loss + ori_loss

    zero_grad_model(r.ego_model)
    train!(r.ego_model, loss)

    return Dict("loc_loss" => Float32(loc_loss), "ori_loss" => Float32(ori_loss),
        "pred_locs" => Array(pred_locs), "pred_yaws" => Array(pred_yaws),
        "locs" => Array(locs), "yaws" => Array(yaws))
end
end