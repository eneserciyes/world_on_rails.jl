module ego_dataset
using NPZ

export EgoDataset

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
            len = length(readdir(fullpath))
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

(d::EgoDataset)(idx::Int) = 
begin 
    episode_idx = d.episode_map[idx]
    index = d.idx_map[idx]

    episode = d.episodes[episode_idx]
    locs = episode[1][index:index+d.T-1, :]
    rots = episode[2][index:index+d.T-1, :]
    spds = episode[3][index:index+d.T-1, :]
    acts = episode[4][index:index+d.T-1, :]
    return locs, rots, spds, acts
end

end