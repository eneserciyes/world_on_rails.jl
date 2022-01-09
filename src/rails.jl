module rails
export RAILS

push!(LOAD_PATH, joinpath(pwd(), "src", "models"))
push!(LOAD_PATH, joinpath(pwd(), "src", "data"))

using Knet
using Knet: Adam, softmax, logsoftmax
using models: set_optim!, zero_grad_model, train!
using Statistics: mean

using ego_model: EgoModel
using main_model: MainModel

using metrics: l1_loss, kl_div

using YAML

struct RAILS
    config::Dict
    device::AbstractString
    ego_model::EgoModel
    main_model::MainModel

    function RAILS(args)
        config = nothing
        open(args["config"], "r") do cfg_file
            config = YAML.load(cfg_file)
        end

        # create ego model
        ego_model = EgoModel(1 / (args["fps"] * (args["num-repeat"] + 1)))
        set_optim!(ego_model, Adam, args["lr"])

        # create main_model
        main_model = MainModel(config)
        set_optim!(main_model, Adam, args["lr"])

        new(config, args["device"], ego_model, main_model)
    end
end

function train_ego(r::RAILS, locs, rots, spds, acts)
    locs = selectdim(locs, length(size(locs)), 1:2)
    yaws = selectdim(rots, length(size(rots)), 3:size(rots)[end]) .* pi ./ 180

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
    ori_loss =
        l1_loss(cos.(pred_yaws), cos.(yaws[:, 2:end])) +
        l1_loss(sin.(pred_yaws), sin.(yaws[:, 2:end]))

    loss = @diff loc_loss + ori_loss

    zero_grad_model(r.ego_model)
    train!(r.ego_model, loss)

    return Dict(
        "loc_loss" => Float32(loc_loss),
        "ori_loss" => Float32(ori_loss),
        "pred_locs" => Array(pred_locs),
        "pred_yaws" => Array(pred_yaws),
        "locs" => Array(locs),
        "yaws" => Array(yaws),
    )
end

function train_main(
    r::RAILS,
    wide_rgbs,
    wide_sems,
    narr_rgbs,
    narr_sems,
    act_vals,
    spds,
    cmds,
)
    wide_rgbs = KnetArray{Float32}(permutedims(Float32.(wide_rgbs), (1, 4, 2, 3)))
    narr_rgbs = KnetArray{Float32}(permutedims(Float32.(narr_rgbs), (1, 4, 2, 3)))
    wide_sems = KnetArray{Int64}(Int64.(wide_sems))
    narr_sems = KnetArray{Int64}(Int64.(narr_sems))

    act_vals = KnetArray{Float32}(permutedims(act_vals, (1, 3, 2, 4)));
    # no change for spds and cmds

    # pass through model
    act_probs = softmax(act_vals./r.config["temperature"], dims = 3); #TODO: check if softmax value is same
    if r.config["use_narr_cam"]
        (act_outputs, wide_seg_outputs, narr_seg_outputs) = r.main_model(
            wide_rgbs,
            narr_rgbs,
            spd = ifelse(r.config["all_speeds"], nothing, spds),
        )
    else
        (act_outputs, wide_seg_outputs) = r.main_model(
            wide_rgbs,
            narr_rgbs,
            spd = ifelse(r.config["all_speeds"], nothing, spds),
        )
    end
    #TODO: test every part correctness with random input.

    if r.config["all_speeds"]
        act_loss = kl_div(logsoftmax(act_outputs, dims=4), act_probs, reduction="none")
        act_loss = mean(act_loss, dims=[3,4])
    else
        act_probs = spd_lerp(act_probs, spds)
        act_loss = kl_div(logsoftmax(act_outputs, dims=3), act_probs, reduction="none")
        act_loss = mean(act_loss, dims=2)
    end

    turn_loss = (act_loss[:,1] + act_loss[:,2] + act_loss[:,3] +  act_loss[:,4]) / 4
    lane_loss = (act_loss[:,5]+act_loss[:,6]+act_loss[:,4])/3
    foll_loss = act_loss[:,4]

    is_turn = (cmds.==0).|(cmds.==1).|(cmds.==2)
    is_lane = (cmds.==4).|(cmds.==5)

    act_loss = mean(@. ifelse(is_turn, turn_loss, foll_loss + @. ifelse(is_lane, lane_loss, foll_loss)))
    seg_loss = 0 # TODO: find how to implement cross-entropy and interpolate

    if r.config["use_narr_cam"]
        seg_loss = seg_loss + 0 #TODO: cross-entropy narr_sems
        seg_loss = seg_loss / 2
    end

    loss = @diff act_loss + r.config["seg_weight"] * seg_loss

    # backpropagate
    zero_grad_model(r.main_model)
    train!(r.main_model, loss)

    # get action probabilities
    #TODO: implement bellman_updater _batch_lerp
    if r.config["all_speeds"]
        act_prob = r.bellman_updater._batch_lerp(permutedims(act_probs[1,Int(cmds[1])+1], (2,1)), spds[1:1], min_val=r.bellman_updater._min_speeds, max_val=r.bellman_updater._max_speeds)
        pred_act_prob = r.bellman_updater._batch_lerp(permutedims(softmax(act_outputs[1,Int(cmds[1])+1], dims=2), (2,1)), spds[1:1],min_val=r.bellman_updater._min_speeds, max_val=r.bellman_updater._max_speeds)
    else
        act_prob = act_probs[1,Int(cmds[1])+1]
        pred_act_prob = softmax(act_outputs[1,Int(cmds[1])+1], dims=1)
    end

    return Dict(
        "act_loss" => Float32(act_loss),
        "seg_loss" => Float32(seg_loss),
        "gt_seg" => Array(wide_sems[1]),
        "pred_seg" => argmax(Array(wide_seg_outputs[1]), dims=1),
        "cmd" => Int(cmds[1]),
        "spds" => Int(spds[1]),
        "wide_rgb" => permutedims(UInt8.(Array(wide_rgbs[1])), (2,3,1)),
        "narr_rgb" => permutedims(UInt8.(Array(narr_rgbs[1])), (2,3,1)),
        "act_prob" => reshape(reverse(Array(act_prob)), (r.config["num_throts"], r.config["num_steers"])),
        "pred_act_prob" => reshape(reverse(Array(pred_act_prob)), (r.config["num_throts"], r.config["num_steers"])),
        "act_brak" => Float32(act_prob[end]),
        "pred_act_brak" => Float32(pred_act_prob[end])
    )

end

function spd_lerp(v, x)
    #TODO: implement spd_lerp
end

end