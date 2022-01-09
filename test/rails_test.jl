using Test
using Revise
using ArgParse

push!(LOAD_PATH, "src")
push!(LOAD_PATH, "src", "models")

using ego_model: EgoModel
using rails: RAILS

function parse_cil()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--resume"
            help = "model file to train on"
            default = false
        "--data-dir"
            help = "Main data dir"
            default = "/home/enes/avg/WoR/world_on_rails.jl/data/main_mini"
        "--config"
            help = "main configuration file"
            default = "/home/enes/avg/WoR/world_on_rails.jl/configs/config.yaml"
        "--fps"
            help = "frame per second"
            default = 20
        "--num-repeat"
            help = "Should be consistent with autoagents/collector_agents/config.yaml"
            default = 4
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
rails = RAILS(args)

@testset "RAILS general tests" begin
    @test isa(rails.ego_model, EgoModel)

end

@testset "train_main tests" begin
    


end


