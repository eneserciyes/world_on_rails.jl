push!(LOAD_PATH, joinpath(dirname(pwd()), "src", "data"));
push!(LOAD_PATH, joinpath(dirname(pwd()), "src"));

using rails: RAILS
using ego_dataset: compile_data

using ArgParse


function main(args)
    rails = RAILS(args)
    data = compile_data(args)
    
    counter = 0
    for epoch = 1:args.num_epoch
        for (locs, rots, spds, acts) = data
            opt_info = rails.train_ego(locs, rots, spds, acts)
            if counter % args.num_per_log == 0
                #logger.log_ego(opt_info) #TODO: implement the logger
            end
            counter += 1
        end
    end 
end

function parse_cil()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--data-dir"
            help = "Ego data dir"
        "--traj-len"
            help = "Trajectory length for ego model"
            default = 10
        "--config"
            help = "main configuration file"
        "--batchsize"
            help = "Batchsize"
            default = 128
        "--shuffle"
            help = "Shuffle the ego dataset"
            action = :store_true
            default = false
        "--device"
            help = "cpu or cuda"
            default = "cuda"
        "--lr"
            help = "learning rate"
            default = 1e-2
        "--num-epoch"
            help = "Number of epochs to train"
            default = 100
        "--num-per-log"
            help = "Frequency to log (step)"
            default=10
    end
    return parse_args(s)
end

args = parse_cil()
main(args)