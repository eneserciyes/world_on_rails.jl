module ego_dataset

using NPZ
using Random: randperm


export EgoDataset, compile_data

struct EgoDataset
    T::Int
    num_frames::Int
    episodes::Array
    episode_map::Dict
    idx_map::Dict

    function EgoDataset(data_dir::AbstractString, T = 10)
        num_frames = 0
        episodes = []
        episode_map = Dict{Int,Int}()
        idx_map = Dict{Int,Int}()

        for fullpath in readdir(data_dir, join = true)
            len = length(readdir(fullpath)) ÷ 4 # folder contains loc,rot,spd,act for each frame
            @show len
            if len < T
                continue
            end

            locs = zeros(len, 3) # location 3-element vector
            rots = zeros(len, 3) # rotation 3-element vector
            spds = zeros(len, 1) # speed 1-element
            acts = zeros(len, 3) # action 3-element vector

            for file in readdir(fullpath, join = true)
                base = basename(file)
                idx = parse(Int, base[end-8:end-4]) + 1
                if startswith(base, "act")
                    acts[idx, :] = npzread(file)
                elseif startswith(base, "loc")
                    locs[idx, :] = npzread(file)
                elseif startswith(base, "rot")
                    rots[idx, :] = npzread(file)
                elseif startswith(base, "spd")
                    spds[idx, 1] = npzread(file)
                end
            end

            offset = num_frames
            for i = 1:len-T+1
                num_frames += 1
                # each index for the current batch maps to the current episode in episodes list.
                # length(episodes)+1 because we will add the episode just in a sec.  
                episode_map[offset+i] = length(episodes) + 1
                idx_map[offset+i] = i
            end
            push!(episodes, (locs, rots, spds, acts))
        end
        new(T, num_frames, episodes, episode_map, idx_map)
    end
end

function Base.length(d::EgoDataset)
    return d.num_frames
end

function Base.getindex(d::EgoDataset, idx::Int) 
    episode_idx = d.episode_map[idx]
    index = d.idx_map[idx]

    episode = d.episodes[episode_idx]
    locs = episode[1][index:index+d.T-1, :]
    rots = episode[2][index:index+d.T-1, :]
    spds = episode[3][index:index+d.T-1, :]
    acts = episode[4][index:index+d.T-1, :]
    return locs, rots, spds, acts
end

mutable struct EgoDatasetLoader
    ego_dataset::EgoDataset
    batchsize::Int
    shuffle::Bool
    imax
    indices

    function EgoDatasetLoader(ego_dataset::EgoDataset, batchsize; shuffle=false)
        imax = length(ego_dataset) - (length(ego_dataset) % batchsize)
        new(ego_dataset, batchsize, shuffle, imax, 1:length(ego_dataset))
    end
end

function Base.iterate(d::EgoDatasetLoader, i=1)
    len = length(d.ego_dataset)
    if i > d.imax
        return nothing
    end

    if d.shuffle && i==1
        d.indices = randperm(len)
    end
    nexti = min(i+d.batchsize-1, len)
    ids = d.indices[i:nexti]
    @show ids

    locs_batch = zeros(d.ego_dataset.T, 3, d.batchsize)
    rots_batch = zeros(d.ego_dataset.T, 3, d.batchsize)
    spds_batch = zeros(d.ego_dataset.T, 1, d.batchsize)
    acts_batch = zeros(d.ego_dataset.T, 3, d.batchsize)

    for idx = ids
        locs, rots, spds, acts = d.ego_dataset[idx]
        batch_idx = (idx-1)%16 + 1
        locs_batch[:,:,batch_idx] = locs
        rots_batch[:,:,batch_idx] = rots
        spds_batch[:,:,batch_idx] = spds
        acts_batch[:,:,batch_idx] = acts
    end
    
    return ((locs_batch, rots_batch, spds_batch, acts_batch), nexti+1)
end


function compile_data(args)
    ego_data = EgoDataset(args.data_dir, args.traj_len)
    return EgoDatasetLoader(ego_data, args.batchsize; shuffle=args.shuffle)
end