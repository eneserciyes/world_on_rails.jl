module rails
export RAILS

push!(LOAD_PATH, joinpath(pwd(), "models"))
push!(LOAD_PATH, joinpath(pwd(), "data"))

using ego_model: EgoModel
using ego_dataset: EgoDataset
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

        new(config, args.device, ego_model)
    end
end

function train_ego(r::RAILS, locs, rots, spds, acts)


end
end