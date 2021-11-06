module Data

using YAML: load_file
using Images
using Images.FileIO
using LMDB
using LMDB: create, open, close, put!, start

struct MainDataset
    config::Dict
    num_frames::Int
    txn_map::Dict
    dbi_map::Dict
    idx_map::Dict
    yaw_map::Dict
    file_map::Dict

    function MainDataset(data_dir::AbstractString, config_path::AbstractString)
        # read the config file
        num_frames = 0
        config = load_file(config_path)
        txn_map = Dict()
        dbi_map = Dict()
        idx_map = Dict()
        yaw_map = Dict()
        file_map = Dict()
        
        for full_path in readdir(data_dir, join=true)
            if !isdir(full_path)
                continue
            end
            env = create()
            open(env, full_path)
            txn = start(env)

            dbi = open(txn)
            
            n = parse(Int64, LMDB.get(txn, dbi, "len", String))
            if n < config["num_plan"]+1
                print(full_path, " is too small. consider deleting")
                close(env, dbi)
            else
                offset = num_frames
                for i in 0:n-config["num_plan"]-1
                    num_frames+=1
                    for j in 0:length(config["camera_yaws"])-1
                        txn_map[(offset+i)*length(config["camera_yaws"])+j] = txn
                        dbi_map[(offset+i)*length(config["camera_yaws"])+j] = dbi
                        idx_map[(offset+i)*length(config["camera_yaws"])+j] = i
                        yaw_map[(offset+i)*length(config["camera_yaws"])+j] = j
                        file_map[(offset+i)*length(config["camera_yaws"])+j] = full_path
                    end
                end
            end
        end
        print(data_dir,": ", num_frames, " frames ", "x ", length(config["camera_yaws"]))
        new(config, num_frames, txn_map, dbi_map, idx_map, yaw_map, file_map)
    end
end

function access(tag::AbstractString, txn::Transaction, dbi::DBI, index::Int, T::Int, dtype::Type; preprocess = x->x)
    preprocess.([LMDB.get(txn, dbi, "$(tag)_$(lpad(t, 5, '0'))", dtype) for t in index:index+(T-1)])
end

function get_item(d::MainDataset, state::Int)
    T = d.config["num_plan"]
    idx = state
    if !d.config["multi_cam"]
        idx *= length(d.config["camera_yaws"])
    end
    
    lmdb_txn = d.txn_map[idx]
    lmdb_dbi = d.dbi_map[idx]
    index = d.idx_map[idx]
    cam_index = d.yaw_map[idx]

    locs = access("loc", lmdb_txn, lmdb_dbi, index, T+1, Vector{Float32})
    rots = access("rot", lmdb_txn, lmdb_dbi, index, T, Vector{Float32})
    spds = access("spd", lmdb_txn, lmdb_dbi, index, T, Vector{Float32})
    
    decode = (x -> FileIO.load(IOBuffer(x)))
    lbls = [access("lbl_$(lpad(l, 2, '0'))", lmdb_txn, lmdb_dbi, index+1, T, Vector{UInt8}; preprocess=decode) for l in 0:11]
    wide_rgb = access("wide_$(cam_index)", lmdb_txn, lmdb_dbi, index, 1, Vector{UInt8}; preprocess=decode)
    wide_sem = access("wide_sem_$(cam_index)", lmdb_txn, lmdb_dbi, index, 1, Vector{UInt8}; preprocess=decode)
    narr_rgb = access("narr_$(cam_index)", lmdb_txn, lmdb_dbi, index, 1, Vector{UInt8}; preprocess=decode)
    
    cmd = access("cmd", lmdb_txn, lmdb_dbi, index, 1, Float32)
    wide_rgb, wide_sem, narr_rgb, lbls, locs, rots, spds, trunc.(Int, cmd)
end

Base.length(d::MainDataset) = (if (d.config["multi_cam"]) d.num_frames*length(d.config["camera_yaws"]) else d.num_frames end)
Base.iterate(d::MainDataset, state=1) = state > length(d) ? nothing : get_item(d, state), state+1 

DATA_ROOT = "/home/enes/avg/WoR/dataset"
CONFIG_PATH = "/home/enes/avg/WoR/WorldOnRails/config.yaml"

end