
function main(args)
    rails = RAILS(args)
    data = compile_data(args)

    if args.resume
        println("Loading checkpoint from $(args.resume)..")
        #TODO: implement model loading and setting start here.
    else
        start = 0
    end

    counter = 0
    for epoch = start:start+args.num_epoch
        for (wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds) = data
            opt_info = rails.train_main(wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds)

            if counter % args.num_per_log == 0
                #logger.log_main_info(counter, opt_info) #TODO: implement logger main
            end
            counter += 1

            if epoch + 1 % args.num_per_save == 0
                save_path = "$(args.save_dir)/main_model_$(epoch+1).th"
                #TODO: implement model saving here.
                print("saved to $(save_path)..")
            end
        end
    end
end

function parse_cil()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--resume"
            help = "model file to train on"
        "--data-dir"
            help = "Main data dir"
        "--config"
            help = "main configuration file"
        "--batchsize"
            help = "Batchsize"
            default = 128
        "--device"
            help = "cpu or cuda"
            default = "cuda"
        "--lr"
            default = 1e-2
        "--weight-decay"
            default = 3e-5
        "--num-epoch"
            help = "Number of epochs to train"
            default = 20
        "--num-per-log"
            help = "per iteration"
            default=100
        "--num-per-save"
            help = "per epoch"
            default=1
    end
    return parse_args(s)
end

args = parse_cil()
main(args)