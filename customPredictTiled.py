import torch


def custom_predict_encoder_sliding_window(predictor, input_image: torch.Tensor):
    device = predictor.device
    patch_size = predictor.configuration_manager.patch_size
    encoder = predictor.network.encoder  # Use only the encoder part of the network
    encoder.to(device)
    encoder.eval()

    # Prepare input shape and output shape
    # input_image: (C, X, Y, Z)
    img_shape = input_image.shape[-3:]  # (X, Y, Z)
#    print("Input image shape:", input_image.shape)
    stride = [int(p * predictor.tile_step_size) for p in patch_size]
    # Compute crop locations
    crop_locations = []
    for x in range(0, img_shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1]):
            for z in range(0, img_shape[2] - patch_size[2] + 1, stride[2]):
                crop_locations.append((x, y, z))
    # Ensure last patch covers the end
    if (img_shape[0] - patch_size[0]) % stride[0] != 0:
        crop_locations += [(img_shape[0] - patch_size[0], y, z) for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1]) for z in range(0, img_shape[2] - patch_size[2] + 1, stride[2])]
    if (img_shape[1] - patch_size[1]) % stride[1] != 0:
        crop_locations += [(x, img_shape[1] - patch_size[1], z) for x in range(0, img_shape[0] - patch_size[0] + 1, stride[0]) for z in range(0, img_shape[2] - patch_size[2] + 1, stride[2])]
    if (img_shape[2] - patch_size[2]) % stride[2] != 0:
        crop_locations += [(x, y, img_shape[2] - patch_size[2]) for x in range(0, img_shape[0] - patch_size[0] + 1, stride[0]) for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1])]
    crop_locations = list(set(crop_locations))  # Remove duplicates

    # Preallocate output volume (find encoder output channels by running one patch)
    sample_patch = input_image[..., crop_locations[0][0]:crop_locations[0][0]+patch_size[0],
                                  crop_locations[0][1]:crop_locations[0][1]+patch_size[1],
                                  crop_locations[0][2]:crop_locations[0][2]+patch_size[2]]
    sample_patch = sample_patch.to(device) #unsqueeze(0).to(device)
    #print("Sample patch shape:", sample_patch.shape)
    with torch.no_grad():
        sample_output = encoder(sample_patch)[-1]  # Get the deepest encoder feature map  
    out_channels = sample_output.shape[1]
    #print("sample output shape:", sample_output.shape)
    # Output volume: (batch=1, out_channels, X, Y, Z)
    full_encoder_output = torch.zeros((out_channels, len(crop_locations)), dtype=sample_output.dtype, device="cuda") #"cpu")
    #norm_map = torch.zeros(img_shape, dtype=torch.float, device="cpu")

    # Run all crops
    idx = 0
    for (x, y, z) in crop_locations:
        patch = input_image[..., x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]].to(device)
        with torch.no_grad():
            patch_out = encoder(patch)[-1][0]  # Get deepest encoder feature map
        # Add to output volume (overlapping locations are averaged)
        #full_encoder_output[:, x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += patch_out
        #norm_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += 1.0
        full_encoder_output[:, idx] = patch_out.mean(dim=(1,2,3)).detach() #.cpu()  # Global average pool the patch output
        idx += 1
    # Normalize overlapping areas
    # norm_map[norm_map == 0] = 1.0  # Prevent division by zero
    # for c in range(out_channels):
    #    full_encoder_output[c] /= norm_map

    return full_encoder_output.max(dim=1).values