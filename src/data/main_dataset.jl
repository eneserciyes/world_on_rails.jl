module main_dataset

export compile_data

using YAML
using JSON

struct MainDataset
    #TODO: main dataset
    T::Int
    camera_yaws
    wide_crop_top
    narr_crop_bottom
    seg_channels
    num_speed
    num_steers
    num_throts
    multi_cam # ablation option
    num_frames

    episode_map::Dict
    idx_map::Dict
    yaw_map::Dict
    file_map::Dict

    function MainDataset(data_dir::AbstractString, config_dir::AbstractString)
        config = YAML.load_file(config_dir)
        num_frames = 0
        
        episode_map = Dict{Int,Int}()
        idx_map = Dict{Int,Int}()
        yaw_map = Dict{Int,Int}()
        file_map = Dict{Int,Int}()

        for fullpath in readdir(data_dir, join = true)
            data_json = JSON.parsefile(joinpath(fullpath, "data.json"))
            n = data_json["len"] - config["num_plan"]
            if n < config["num_plan"] + 1
                print("$(fullpath) is too small. Consider deleting it..")
            else
                offset = num_frames
                for i = 1:n-config["num_plan"]
                    num_frames += 1
                    for j = 1:length(config["camera_yaws"])
                        episode_map[(offset+i-1)*length(config["camera_yaws"]) + j] = fullpath
                    end
                end
            end

        new(
            config["num_plan"],
            config["camera_yaws"],
            config["wide_crop_top"],
            config["narr_crop_bottom"],
            config["seg_channels"],
            config["num_speeds"],
            config["num_steers"],
            config["num_throts"],
            config["multi_cam"]
            num_frames
        )
    end

end

struct MainDatasetLoader
    #TODO: main dataset loader
end

function compile_data()
    #TODO: complete compile data
end

end