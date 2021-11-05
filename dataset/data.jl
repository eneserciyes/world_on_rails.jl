module Data

using YAML: load_file
using Images
using LMDB
using LMDB: create, open, close, put!, start

struct MainDataset
    config::Dict
    num_frames::Int
    txn_map::Dict
    idx_map::Dict
    yaw_map::Dict
    file_map::Dict

    function MainDataset(data_dir::AbstractString, config_path::AbstractString)
        # read the config file
        num_frames = 0
        config = load_file(config_path)
        txn_map = Dict()
        idx_map = Dict()
        yaw_map = Dict()
        file_map = Dict()
        
        for full_path in readdir(data_dir, join=true)
            @show full_path
            if !isdir(full_path)
                continue
            end
            env = create()
            open(env, full_path)
            txn = start(env)

            dbi = open(txn)
            
            n = parse(Int64, LMDB.get(txn, dbi, "len", String))
            @show n
            if n < config["num_plan"]+1
                print(full_path, " is too small. consider deleting")
                close(env, dbi)
            else
                offset = num_frames
                for i in 1:n-config["num_plan"]
                    num_frames+=1
                    for j in 1:length(config["camera_yaws"])
                        txn_map[(offset+i)*length(config["camera_yaws"])+j] = txn
                        idx_map[(offset+i)*length(config["camera_yaws"])+j] = i
                        yaw_map[(offset+i)*length(config["camera_yaws"])+j] = j
                        file_map[(offset+i)*length(config["camera_yaws"])+j] = full_path
                    end
                end
            end
        end
        print(data_dir,": ", num_frames, " frames ", "x ", length(config["camera_yaws"]))
    end
end

DATA_ROOT = "/home/enes/avg/WoR/dataset"
CONFIG_PATH = "/home/enes/avg/WoR/WorldOnRails/config.yaml"
MainDataset(DATA_ROOT, CONFIG_PATH)

end