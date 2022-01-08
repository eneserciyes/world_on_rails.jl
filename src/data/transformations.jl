function to_tensor(x)
    return permutedims(UInt8.((x |> channelview) .* 255), (2,3,1))
end

function to_img(x)
    return colorview(RGB, permutedims(Float32.(x) ./255, (3,1,2)))
end