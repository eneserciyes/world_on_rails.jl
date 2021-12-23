push!(LOAD_PATH, joinpath(dirname(pwd()), "src", "models"));
push!(LOAD_PATH, joinpath(dirname(pwd()), "src", "data"));
push!(LOAD_PATH, joinpath(dirname(pwd()), "src"));

using rails: RAILS
using ego_model: EgoModel
using ego_dataset: EgoDataset

using ArgParse

function parse_cil()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--data-dir"
            help = "Ego data dir"
        "--config"
            help = "main configuration file"
        "--device"
            help = "cpu or cuda"
    end
    return parse_args(s)
end

function main()
    args = parse_cil()
    rails = RAILS(args)
    ego_data = EgoDataset(args.data_dir) #TODO: implement a dataloader instead of dataset

end

main()