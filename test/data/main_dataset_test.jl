using Test
using Revise

push!(LOAD_PATH, joinpath("src", "data"))

import main_dataset:LabeledMainDataset


@testset "Main Dataset tests" begin
    using PyCall
    using Knet:KnetArray
    using Statistics:mean

    main_dataset = LabeledMainDataset("/home/enes/avg/WoR/world_on_rails.jl/data/main_mini",
     "/home/enes/avg/WoR/world_on_rails.jl/configs/config.yaml")
    println("Starting Main Dataset Tests...")

    # test augmenter pyobject
    @test isa(main_dataset.augmenter, PyObject)

    # test length
    @test length(main_dataset) == 516

    # test rgb image reading
    (wide_rgb, wide_sem, narr_rgb, narr_sem, act_val, spd, cmd) = main_dataset[1]

    # test rgb image mean
    @test floor(mean(wide_rgb)) == 31.0

    # test image type and size
    @test isa(wide_rgb, Array{UInt8})
    @test isa(narr_rgb, Array{UInt8})
    @test size(wide_rgb) == (240 - main_dataset.wide_crop_top + 1,480,3)
    
end

@testset "PyCall tests" begin
    using PyCall; using Images
    # test py string 
    py"""
    import numpy as np

    def sin(x):
        return np.sin(x)
    """
    @test py"sin"(0) == 0.0

    # test pyinclude macro
    @pyinclude("test/data/pycall_test/sin.py")
    @test py"sin"(0) == 0.0

    # test augment.py pyinclude macro
    @pyinclude("src/data/augment.py")
    @test isa(py"augment"(), Vector{PyObject}) == true

    # test pycall array conversion inhibition
    @pyinclude("src/data/augment.py")
    augmenter = pycall(py"augment", PyObject)
    @test isa(augmenter, PyObject)
    @test augmenter.random_order == true

end
