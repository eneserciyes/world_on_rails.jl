module main_dataset

export compile_data, LabeledMainDataset

using YAML
using JSON

using Images
using Images.FileIO

using Knet: KnetArray

struct LabeledMainDataset
    #TODO: main dataset
    T::Int
    camera_yaws
    wide_crop_top
    narr_crop_bottom
    seg_channels
    num_speeds
    num_steers
    num_throts
    multi_cam # ablation option
    num_frames

    idx_map::Dict
    yaw_map::Dict
    json_map::Dict
    path_map::Dict

    function LabeledMainDataset(data_dir::AbstractString, config_dir::AbstractString)
        config = YAML.load_file(config_dir)
        num_frames = 0

        idx_map = Dict{Int,Int}()
        yaw_map = Dict{Int,Int}()
        json_map = Dict{Int,Dict}()
        path_map = Dict{Int,String}()

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
                        idx_map[(offset+i-1)*length(config["camera_yaws"])+j] = i
                        yaw_map[(offset+i-1)*length(config["camera_yaws"])+j] = j
                        json_map[(offset+i-1)*length(config["camera_yaws"])+j] = data_json
                        path_map[(offset+i-1)*length(config["camera_yaws"])+j] = fullpath
                    end
                end
            end
        end

        println("$(data_dir): $(num_frames) frames (x$(length(config["camera_yaws"])))")

        new(
            config["num_plan"],
            config["camera_yaws"],
            config["wide_crop_top"],
            config["narr_crop_bottom"],
            config["seg_channels"],
            config["num_speeds"],
            config["num_steers"],
            config["num_throts"],
            config["multi_cam"],
            num_frames,
            idx_map,
            yaw_map,
            json_map,
            path_map
        )
    end
end

function Base.length(d::LabeledMainDataset)
    return d.num_frames
end

function load_img(path::AbstractString)
    img = FileIO.load(path)
    return permuteddimsview(img |> channelview, (2, 3, 1)) # (H,W, Channel) shaped 0-1 interval image. 
end

semantic_mapping = [
    ([220, 20, 60], 4), # pedestrian
    ([157, 234, 50], 6), # road line
    ([128, 64, 128], 7), # Road
    ([244, 35, 232], 8), # SideWalk
    ([0, 0, 142], 10), # Vehicles
    ([250, 170, 30], 18) # Traffic Light
]

function read_sem(path::AbstractString, filter_classes::Vector)
    segimg = load_img(path) * 255
    segimg_con = zeros(Float32, size(segimg)[1:2])

    for (i,(key,value)) in enumerate(semantic_mapping)
        if (value in filter_classes)
            mask = (segimg[:,:,1] .== key[1]) .& (segimg[:,:,2] .== key[2]) .& (segimg[:,:,3] .== key[3])
            segimg_con[mask] .= i
        end
    end
    return segimg_con
end

function augment(img::Array)
    #TODO: use PyCall to call preprocessing here. 
end

function Base.getindex(d::LabeledMainDataset, idx::Int)
    if !d.multi_cam
        idx *= length(d.camera_yaws)
    end

    index = d.idx_map[idx] - 1 # json idxs start from zero
    cam_index = d.yaw_map[idx] - 1 # cam idxs in json start from zero
    data_json = d.json_map[idx]
    path = d.path_map[idx]

    wide_rgb = load_img(joinpath(path, "rgbs", "wide_$(cam_index)_$(lpad(index,5,"0")).jpg"))
    wide_sem = read_sem(joinpath(path, "rgbs", "wide_sem_$(cam_index)_$(lpad(index,5,"0")).png"), d.seg_channels)
    narr_rgb = load_img(joinpath(path, "rgbs", "narr_$(cam_index)_$(lpad(index,5,"0")).jpg"))
    narr_sem = read_sem(joinpath(path, "rgbs", "narr_sem_$(cam_index)_$(lpad(index,5,"0")).png"), d.seg_channels)

    cmd = data_json[string(index)]["cmd"]
    spd = data_json[string(index)]["spd"]

    act_val = reshape(data_json[string(index)]["act$(cam_index)"], (d.num_speeds, d.num_steers * d.num_throts + 1, 6))
    act_val = permutedims(act_val, (2,1,3)) # permute dims because julia is column-major

    # Crop cameras
    wide_rgb = wide_rgb[d.wide_crop_top:end, :, :] #TODO: check if reversing required here.
    wide_sem = wide_sem[d.wide_crop_top:end,:]
    narr_rgb = narr_rgb[begin:end-d.narr_crop_bottom, :, :] #TODO: check if reversing required here.
    narr_sem = narr_sem[begin:end-d.narr_crop_bottom,:]

    #Augment
    # wide_rgb = augment(wide_rgb)
    # narr_rgb = augment(narr_rgb)

    return (KnetArray{Float32}(wide_rgb), KnetArray{Float32}(wide_sem),
        KnetArray{Float32}(narr_rgb), KnetArray{Float32}(narr_sem),
        KnetArray{Float32}(act_val), Float32(spd[1]), Float32(cmd[1]))

end

struct LabeledMainDatasetLoader
    #TODO: main dataset loader
end

function compile_data()
    #TODO: complete compile data
end

end