#source https://github.com/RyannDaGreat/CommonSource/blob/master/noise_warp.py
# https://github.com/RyannDaGreat/rp

import torch.nn.functional as F

import numpy as np
import torch
from einops import rearrange

def unique_pixels(image):
    """
    Find unique pixel values in an image tensor and return their RGB values, counts, and inverse indices.

    Args:
        image (torch.Tensor): Image tensor of shape [c, h, w], where c is the number of channels (e.g., 3 for RGB),
                              h is the height, and w is the width of the image.

    Returns:
        tuple: A tuple containing three tensors:
            - unique_colors (torch.Tensor): Tensor of shape [u, c] representing the unique RGB values found in the image,
                                            where u is the number of unique colors.
            - counts (torch.Tensor): Tensor of shape [u] representing the counts of each unique color.
            - index_matrix (torch.Tensor): Tensor of shape [h, w] representing the inverse indices of each pixel,
                                           mapping each pixel to its corresponding unique color index.
    """
    c, h, w = image.shape

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Find unique RGB values, counts, and inverse indices
    unique_colors, inverse_indices, counts = torch.unique(flattened_pixels, dim=0, return_inverse=True, return_counts=True, sorted=False)
    # unique_colors, inverse_indices, counts = torch.unique_consecutive(flattened_pixels, dim=0, return_inverse=True, return_counts=True)

    # Get the number of unique indices
    u = unique_colors.shape[0]

    # Reshape the inverse indices back to the original image dimensions [h, w] using einops
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)

    # Assert the shapes of the output tensors
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    assert index_matrix.min() == 0
    assert index_matrix.max() == u - 1

    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    """
    Sum the values in the CHW image tensor based on the indices specified in the HW index matrix.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W], where C is the number of channels,
                              H is the height, and W is the width of the image.
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique value.
                                     Indices range [0, U), where U is the number of unique indices

    Returns:
        torch.Tensor: Tensor of shape [U, C] representing the sum of values in the image tensor
                      based on the indices in the index matrix, where U is the number of unique
                      indices in the index matrix.
    """
    c, h, w = image.shape
    u = index_matrix.max() + 1

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Create an output tensor of shape [u, c] initialized with zeros
    output = torch.zeros((u, c), dtype=flattened_pixels.dtype, device=flattened_pixels.device)

    # Scatter sum the flattened pixel values using the index matrix
    output.index_add_(0, index_matrix.view(-1), flattened_pixels)

    # Assert the shapes of the input and output tensors
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"
    assert index_matrix.shape == (h, w), f"Expected index_matrix shape: ({h}, {w}), but got: {index_matrix.shape}"
    assert output.shape == (u, c), f"Expected output shape: ({u}, {c}), but got: {output.shape}"

    return output

def indexed_to_image(index_matrix, unique_colors):
    """
    Create a CHW image tensor from an HW index matrix and a UC unique_colors matrix.

    Args:
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique color.
        unique_colors (torch.Tensor): Unique colors matrix tensor of shape [U, C] containing
                                      the unique color values, where U is the number of unique
                                      colors and C is the number of channels.

    Returns:
        torch.Tensor: Image tensor of shape [C, H, W] representing the reconstructed image
                      based on the index matrix and unique colors matrix.
    """
    h, w = index_matrix.shape
    u, c = unique_colors.shape

    # Assert the shapes of the input tensors
    assert index_matrix.max() < u, f"Index matrix contains indices ({index_matrix.max()}) greater than the number of unique colors ({u})"

    # Gather the colors based on the index matrix
    flattened_image = unique_colors[index_matrix.view(-1)]

    # Reshape the flattened image to [h, w, c]
    image = rearrange(flattened_image, "(h w) c -> h w c", h=h, w=w)

    # Rearrange the image tensor from [h, w, c] to [c, h, w] using einops
    image = rearrange(image, "h w c -> c h w")

    # Assert the shape of the output tensor
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"

    return image
    
def calculate_wave_pattern(h, w, frame):
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    
    # Calculate the distance from the center of the image
    center_x, center_y = w // 2, h // 2
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate the angle from the center of the image
    angle_from_center = torch.atan2(y - center_y, x - center_x)
    
    # Calculate the wave pattern based on the distance and angle
    wave_freq = 0.05  # Frequency of the waves
    wave_amp = 10.0   # Amplitude of the waves
    wave_offset = frame * 0.05  # Offset for animation
    
    dx = wave_amp * torch.cos(dist_from_center * wave_freq + angle_from_center + wave_offset)
    dy = wave_amp * torch.sin(dist_from_center * wave_freq + angle_from_center + wave_offset)
    
    return dx, dy

def starfield_zoom(h, w, frame):
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    
    # Calculate the distance from the center of the image
    center_x, center_y = w // 2, h // 2
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate the angle from the center of the image
    angle_from_center = torch.atan2(y - center_y, x - center_x)
    
    # Calculate the starfield zoom effect
    zoom_speed = 0.01  # Speed of the zoom effect
    zoom_scale = 1.0 + frame * zoom_speed  # Scale factor for the zoom effect
    
    # Calculate the displacement based on the distance and angle
    dx = dist_from_center * torch.cos(angle_from_center) / zoom_scale
    dy = dist_from_center * torch.sin(angle_from_center) / zoom_scale
    
    return dx, dy

_arange_cache={}
def _cached_arange(length, device, dtype):
    code=hash((length,device,dtype))
    if code in _arange_cache:
        return _arange_cache[code]

    
    _arange_cache[code]= torch.arange(length , device=device, dtype=dtype)
    return _arange_cache[code]

def fast_nearest_torch_remap_image(image, x, y, *, relative=False, add_alpha_mask=False, use_cached_meshgrid=False):
    # assert rp.r.is_torch_image(image), "image must be a torch tensor with shape [C, H, W]"
    # assert is_torch_tensor(x) and is_a_matrix(x), "x must be a torch tensor with shape [H_out, W_out]"
    # assert is_torch_tensor(y) and is_a_matrix(y), "y must be a torch tensor with shape [H_out, W_out]"
    # assert x.shape == y.shape, "x and y must have the same shape, but got x.shape={} and y.shape={}".format(x.shape, y.shape)
    # assert image.device==x.device==y.device, "all inputs must be on the same device"

    # pip_import('torch')

    import torch

    in_c, in_height, in_width = image.shape
    out_height, out_width = x.shape

    if add_alpha_mask:
        alpha_mask = torch.ones_like(image[:1])
        image = torch.cat([image, alpha_mask], dim=0)

    if torch.is_floating_point(x): x = x.round_().long()
    if torch.is_floating_point(y): y = y.round_().long()

    if relative:
        # assert in_height == out_height, "For relative warping, input and output heights must match, but got in_height={} and out_height={}".format(in_height, out_height)
        # assert in_width  == out_width , "For relative warping, input and output widths must match, but got in_width={} and out_width={}".format(in_width, out_width)
        x += _cached_arange(in_width , device=x.device, dtype=x.dtype)
        y += _cached_arange(in_height, device=y.device, dtype=y.dtype)[:,None]

    x.clamp_(0, in_width - 1)
    y.clamp_(0,in_height-1)
    out = image[:, y, x]

    expected_c = in_c+1 if add_alpha_mask else in_c
    assert out.shape == (expected_c, out_height, out_width), "Expected output shape: ({}, {}, {}), but got: {}".format(expected_c, out_height, out_width, out.shape)

    return out


def warp_noise(noise, dx, dy, s=1):
    #This is *certainly* imperfect. We need to have particle swarm in addition to this.

    dx=dx.round_().int()
    dy=dy.round_().int()

    c, h, w = noise.shape
    assert dx.shape==(h,w)
    assert dy.shape==(h,w)

    #s is scaling factor
    hs = h * s
    ws = w * s
    
    #Upscale the warping with linear interpolation. Also scale it appropriately.
    if s!=1:
        up_dx = F.interpolate(dx[None], (hs, ws), interp="bilinear")[0]
        up_dy = F.interpolate(dy[None], (hs, ws), interp="bilinear")[0]
        up_dx *= s
        up_dy *= s

        up_noise = F.interpolate(noise, (hs, ws), interp="nearest")
    else:
        up_dx = dx
        up_dy = dy
        up_noise = noise
    assert up_noise.shape == (c, hs, ws)

    # Warp the noise - and put 0 where it lands out-of-bounds
    # up_noise = rp.torch_remap_image(up_noise, up_dx, up_dy, relative=True, interp="nearest")
    up_noise = fast_nearest_torch_remap_image(up_noise, up_dx, up_dy, relative=True)
    assert up_noise.shape == (c, hs, ws)
    
    # Regaussianize the noise
    output, _ = regaussianize(up_noise)

    #Now we resample the noise back down again
    if s!=1:
        output = F.interpolate(output, (h, w), interp='area')
        output = output * s #Adjust variance by multiplying by sqrt of area, aka sqrt(s*s)=s

    return output


def regaussianize(noise):
    c, hs, ws = noise.shape

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(noise[:1])
    u = len(unique_colors)
    assert unique_colors.shape == (u, 1)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (hs, ws)

    foreign_noise = torch.randn_like(noise)
    assert foreign_noise.shape == noise.shape == (c, hs, ws)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, hs, ws)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    assert zeroed_foreign_noise.shape == (c, hs, ws)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, hs, ws)

    #To upsample noise, we must first divide by the area then add zero-sum-noise
    output = noise
    output = output / counts_image ** .5
    output = output + zeroed_foreign_noise

    assert output.shape == noise.shape == (c, hs, ws)

    return output, counts_image
    

def _xy_meshgrid(h,w,device,dtype):
    y, x = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
    )

    output = torch.stack(
        [x, y],
    ).to(device, dtype)

    assert output.shape == (2, h, w)
    return output

def xy_meshgrid_like_image(image):
    """
    Example:
        >>> image=load_image('https://picsum.photos/id/28/367/267')
        ... image=as_torch_image(image)
        ... xy=xy_meshgrid_like_image(image)
        ... display_image(full_range(as_numpy_array(xy[0])))
        ... display_image(full_range(as_numpy_array(xy[1])))
    """
    assert image.ndim == 3, "image is in CHW form"
    c, h, w = image.shape
    return _xy_meshgrid(h,w,image.device,image.dtype)

def noise_to_xyωc(noise):
    assert noise.ndim == 3, "noise is in CHW form"
    zeros=torch.zeros_like(noise[0][None])
    ones =torch.ones_like (noise[0][None])

    #Prepend [dx=0, dy=0, weights=1] channels
    output=torch.concat([zeros, zeros, ones, noise])
    return output

def xyωc_to_noise(xyωc):
    assert xyωc.ndim == 3, "xyωc is in [ω x y c]·h·w form"
    assert xyωc.shape[0]>3, 'xyωc should have at least one noise channel'
    noise=xyωc[3:]
    return noise

def warp_xyωc(
    I,
    F,
    xy_mode="none",
    # USED FOR ABLATIONS:
    expand_only=False,
):
    """
    For ablations, set:
        - expand_only=True #No contraction
        - expand_only='bilinear' #Bilinear Interpolation
        - expand_only='nearest' #Nearest Neighbors Warping
    """
    #Input assertions
    assert F.device==I.device
    assert F.ndim==3, str(F.shape)+' F stands for flow, and its in [x y]·h·w form'
    assert I.ndim==3, str(I.shape)+' I stands for input, in [ω x y c]·h·w form where ω=weights, x and y are offsets, and c is num noise channels'
    xyωc, h, w = I.shape
    assert F.shape==(2,h,w) # Should be [x y]·h·w
    device=I.device
    
    #How I'm going to address the different channels:
    x   = 0        #          // index of Δx channel
    y   = 1        #          // index of Δy channel
    xy  = 2        # I[:xy]
    xyω = 3        # I[:xyω]
    ω   = 2        # I[ω]     // index of weight channel
    c   = xyωc-xyω # I[-c:]   // num noise channels
    ωc  = xyωc-xy  # I[-ωc:]
    # h_dim = 1
    w_dim = 2
    assert c, 'I has no noise channels. There is nothing to warp.'
    assert (I[ω]>0).all(), 'All weights should be greater than 0'

    #Compute the grid of xy indices
    grid = xy_meshgrid_like_image(I)
    assert grid.shape==(2,h,w) # Shape is [x y]·h·w

    #The default values we initialize to. Todo: cache this.
    init = torch.empty_like(I)
    init[:xy]=0
    init[ω]=1
    init[-c:]=0

    #Caluclate initial pre-expand
    pre_expand = torch.empty_like(I)

    #The original plan was to use init xy during expand, because the query position is arbitrary....
    #It doesn't actually make deep sense to copy the offsets during this step, but it doesn't seem to hurt either...
    #BUT I think I got slightly better results...?...so I'm going to do it anyway.
    # pre_expand[:xy] = init[:xy] # <---- Original algorithm I wrote on paper

    #ABLATION STUFF IN THIS PARAGRAPH
    #Using F_index instead of F so we can use ablations like bilinear, bicubic etc
    interp = 'nearest' if not isinstance(expand_only, str) else expand_only
    regauss = not isinstance(expand_only, str)
    F_index = F
    if interp=='nearest':
        #Default behaviour, ablations or not
        F_index=F_index.round()

    pre_expand[:xy] = torch_remap_image(I[:xy], * -F, relative=True, interp=interp)# <---- Last minute change
    pre_expand[-ωc:] = torch_remap_image(I[-ωc:], * -F, relative=True, interp=interp)
    pre_expand[ω][pre_expand[ω]==0]=1 #Give new noise regions a weight of 1 - effectively setting it to init there

    if expand_only:
        if regauss:
            #This is an ablation option - simple warp + regaussianize
            #Enable to preview expansion-only noise warping
            #The default behaviour! My algo!
            pre_expand[-c:]=regaussianize(pre_expand[-c:])[0]
        else:
            #Turn zeroes to noise
            pre_expand[-c:]=torch.randn_like(pre_expand[-c:]) * (pre_expand[-c:]==0) + pre_expand[-c:]
        return pre_expand

    #Calculate initial pre-shrink
    pre_shrink = I.clone()
    pre_shrink[:xy] += F

    #Pre-Shrink mask - discard out-of-bounds pixels
    pos = (grid + pre_shrink[:xy]).round()
    in_bounds = (0<= pos[x]) & (pos[x] < w) & (0<= pos[y]) & (pos[y] < h)
    in_bounds = in_bounds[None] #Match the shape of the input
    out_of_bounds = ~in_bounds
    assert out_of_bounds.dtype==torch.bool
    assert out_of_bounds.shape==(1,h,w)
    assert pre_shrink.shape == init.shape
    pre_shrink = torch.where(out_of_bounds, init, pre_shrink)

    #Deal with shrink positions offsets
    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy

    #FLOATING POINT POSITIONS: I will disable this for now. It does in fact increase sensitivity! But it also makes it less long-term coherent
    assert xy_mode in ['float', 'none'] or isinstance(xy_mode, int)
    if xy_mode=='none':
        pre_shrink[:xy] = 0 #DEBUG: Uncomment to ablate floating-point swarm positions

    if isinstance(xy_mode, int):
        # XY quantization: best to use odd numbers!
        quant = xy_mode
        pre_shrink[:xy] = (
            pre_shrink[:xy] * quant
        ).round() / quant  

    #OTHER ways I tried reducing sensitivity to motion. They work - but 0 is best. Let's just use high resolution.
    # pre_shrink[:xy][pre_shrink[:xy].abs()<.1] = 0  #DEBUG: Uncomment to ablate floating-point swarm positions
    # pre_shrink[:xy] *= -1 #I can't even tell that this is wrong.....
    # pre_shrink[:xy] *= .9 
    # sensitivity_factor = 4

    scat = lambda tensor: torch_scatter_add_image(tensor, *scat_xy, relative=True)

    #Where mask==True, we output shrink. Where mask==0, we output expand.
    shrink_mask = torch.ones(1,h,w,dtype=bool,device=device) #The purpose is to get zeroes where no element is used
    shrink_mask = scat(shrink_mask)
    assert shrink_mask.dtype==torch.bool, 'If this fails we gotta convert it with mask.=astype(bool)'

    # rp.cv_imshow(rp.tiled_images([out_of_bounds[0],shrink_mask[0]]),label='OOB') ; return I #DEBUG - uncomment to see the masks

    #Remove the expansion points where we'll use shrink
    pre_expand = torch.where(shrink_mask, init, pre_expand)
    # rp.cv_imshow(pre_expand[-c:]/5+.5,'preex')

    #Horizontally Concat
    concat_dim = w_dim
    concat     = torch.concat([pre_shrink, pre_expand], dim=concat_dim)

    #Regaussianize
    concat[-c:], counts_image = regaussianize(concat[-c:])
    assert  counts_image.shape == (1, h, 2*w)
    # rp.cv_imshow(concat[-c:]/5+.5,label='regauss') ; return pre_expand #DEBUG - Uncomment to preview regaussianization

    #Distribute Weights
    concat[ω] /= counts_image[0]
    concat[ω] = concat[ω].nan_to_num() #We shouldn't need this, this is a crutch. Final mask should take care of this.

    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=concat_dim)
    assert pre_shrink.shape == expand.shape == (3+c, h, w)
 
    shrink = torch.empty_like(pre_shrink)
    shrink[ω]   = scat(pre_shrink[ω][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy]*pre_shrink[ω][None]) / shrink[ω][None]
    shrink[-c:] = scat(pre_shrink[-c:]*pre_shrink[ω][None]) / scat(pre_shrink[ω][None]**2).sqrt()

    output = torch.where(shrink_mask, shrink, expand)
    output[ω] = output[ω] / output[ω].mean() #Don't let them get too big or too small
    ε = .00001
    output[ω] += ε #Don't let it go too low
    
    # rp.debug_comment([output[ω].min(),output[ω].max()])# --> [tensor(0.0010), tensor(2.7004)]
    # rp.debug_comment([shrink[ω].min(),shrink[ω].max()])# --> [tensor(0.), tensor(2.7004)]
    # rp.debug_comment([expand[ω].min(),expand[ω].max()])# --> [tensor(0.0001), tensor(0.3892)]
    # rp.cv_imshow(rp.apply_colormap_to_image(output[ω]/output[ω].mean()/4),label='weight')
    # rp.cv_imshow(rp.apply_colormap_to_image(output[ω]/10),label='weight')
    assert (output[ω]>0).all()
    # print(end='\r%.08f %.08f'%(float(output[ω].min()), float(output[ω].max())))

    output[ω] **= .9999 #Make it tend towards 1


    return output


class NoiseWarper:
    def __init__(
        self,
        c, h, w,
        device,
        dtype=torch.float32,
        scale_factor=1,
        post_noise_alpha = 0,
        progressive_noise_alpha = 0,
        warp_kwargs=dict(),
    ):

        #Some non-exhaustive input assertions
        assert isinstance(c,int) and c>0
        assert isinstance(h,int) and h>0
        assert isinstance(w,int) and w>0
        assert isinstance(scale_factor,int) and w>=1

        #Record arguments
        self.c=c
        self.h=h
        self.w=w
        self.device=device
        self.dtype=dtype
        self.scale_factor=scale_factor
        self.progressive_noise_alpha=progressive_noise_alpha
        self.post_noise_alpha=post_noise_alpha
        self.warp_kwargs=warp_kwargs

        #Initialize the state
        self._state = self._noise_to_state(
            noise=torch.randn(
                c,
                h * scale_factor,
                w * scale_factor,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def noise(self):
        #TODO: The noise should be downsampled to respect the weights!! 
        noise = self._state_to_noise(self._state)
        weights = self._state[2][None] #xyωc
        noise = (
            F.interpolate((noise * weights).unsqueeze(0), size=(self.h, self.w), mode='area')
            / F.interpolate((weights**2).unsqueeze(0), size=(self.h, self.w), mode='area').sqrt()
        )
        noise = noise * self.scale_factor

        if self.post_noise_alpha:
            noise = mix_new_noise(noise, self.post_noise_alpha)

        return noise

    def __call__(self, dx, dy):

        if isinstance(dx, np.ndarray): dx = torch.tensor(dx).to(self.device, self.dtype)
        if isinstance(dy, np.ndarray): dy = torch.tensor(dy).to(self.device, self.dtype)

        flow = torch.stack([dx, dy]).to(self.device, self.dtype)
        _, oflowh, ofloww = flow.shape #Original height and width of the flow
        
        assert flow.ndim == 3 and flow.shape[0] == 2, "Flow is in [x y]·h·w form"
        flow = F.interpolate(flow.unsqueeze(0), size=(self.h * self.scale_factor, self.w * self.scale_factor)).squeeze(0)

        _, flowh, floww = flow.shape

        #Multiply the flow values by the size change
        flow[0] *= flowh / oflowh * self.scale_factor
        flow[1] *= floww / ofloww * self.scale_factor

        self._state = self._warp_state(self._state, flow)
        return self

    #The following three methods can be overridden in subclasses:

    @staticmethod
    def _noise_to_state(noise):
        return noise_to_xyωc(noise)

    @staticmethod
    def _state_to_noise(state):
        return xyωc_to_noise(state)

    def _warp_state(self, state, flow):

        if self.progressive_noise_alpha:
            state[3:] = mix_new_noise(state[3:], self.progressive_noise_alpha)

        return warp_xyωc(state, flow, **self.warp_kwargs)
    
def blend_noise(noise_background, noise_foreground, alpha):
    """ Variance-preserving blend """
    return (noise_foreground * alpha + noise_background * (1-alpha))/(alpha ** 2 + (1-alpha) ** 2)**.5

def mix_new_noise(noise, alpha):
    """As alpha --> 1, noise is destroyed"""
    if isinstance(noise, torch.Tensor): return blend_noise(noise, torch.randn_like(noise)      , alpha)
    elif isinstance(noise, np.ndarray): return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else: raise TypeError(f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")

def resize_noise(noise, size, alpha=None):
    """
    Can resize gaussian noise, adjusting for variance and preventing cross-correlation
    """
    assert noise.ndim == 3, "resize_noise: noise should be a CHW tensor"
    num_channels, old_height, old_width = noise.shape

    if noise.ndim==4:
        #If given a batch of noises, do it for each one
        return torch.stack([resize_noise(x, new_height, new_width) for x in noise])

    if isinstance(size, int):
        new_height, new_width = int(old_height * size), int(old_width * size)
    else:
        new_height, new_width = size

    assert new_height<=old_height, 'resize_noise: Only useful for shrinking noise, not growing it'
    assert new_width <=old_width , 'resize_noise: Only useful for shrinking noise, not growing it'
    
    grid_y = torch.linspace(0, old_height - 1, steps=new_height)
    grid_x = torch.linspace(0, old_width - 1, steps=new_width)
    y, x = torch.meshgrid(grid_y, grid_x, indexing='ij')

    if alpha is not None:
        #Prepend the alpha
        assert alpha.ndim==2,alpha.shape
        assert alpha.shape==noise.shape[1:],(alpha.shape,noise.shape)
        noise=torch.cat((alpha[None],noise))
        
    resized = torch_scatter_add_image(
        noise,
        x,
        y,
        height=new_height,
        width=new_width,
        interp='floor',
        prepend_ones=alpha is None
    )
    
    total, resized = resized[:1], resized[1:]

    adjusted = resized / total**.5

    return adjusted

def torch_remap_image(image, x, y, *, relative=False, interp='bilinear', add_alpha_mask=False, use_cached_meshgrid=False):
    """
    Remap an image tensor using the given x and y coordinate tensors.
    Out-of-bounds regions will be given 0's
    Analagous to rp.cv_remap_image, which is used for images as defined by rp.is_image()
    
    If the image is RGBA, then out-of-bounds regions will have 0-alpha.
    This is like a UV mapping - where x and y's values are mapped to the image.
    If relative=True, it will warp the image - treating x and y like dx and dy.
        Note: Because this is a mapping, the direction of movement will be opposite dx and dy - so you may need to negate them!

    If add_alpha_mask=True, an additional alpha channel full of 1's will be concatenated to the input image tensor.
    This alpha channel will become 0 for out-of-bounds regions after remapping, serving as an indicator for invalid regions.

    Args:
        image (torch.Tensor): Input image tensor of shape [C, H, W], where C is the number of channels (e.g., 3 for RGB, 4 for RGBA),
                              H is the height, and W is the width of the image.
        x (torch.Tensor): X-coordinate tensor of shape [H_out, W_out] specifying the x-coordinates for remapping,
                          where H_out is the output height and W_out is the output width.
        y (torch.Tensor): Y-coordinate tensor of shape [H_out, W_out] specifying the y-coordinates for remapping.
        relative (bool, optional): If True, treat x and y as deltas (dx and dy) and perform relative warping. Default is False.
        interp (str, optional): Interpolation method. Can be one of 'bilinear', 'bicubic', or 'nearest'. Default is 'bilinear'.
        add_alpha_mask (bool, optional): If True, an additional alpha channel full of 1's will be concatenated to the input image tensor. Default is False.

    Returns:
        torch.Tensor: Remapped image tensor.
            - If add_alpha_mask=False, the output shape is [C, H_out, W_out], where:
                - C is the number of channels in the input image.
                - H_out is the output height.
                - W_out is the output width.
            - If add_alpha_mask=True, the output shape is [C+1, H_out, W_out], where:
                - C+1 includes the additional alpha channel.
                - H_out is the output height.
                - W_out is the output width.

    EXAMPLE:
        >>> def calculate_wave_pattern(h, w, frame):
        ...     # Create a grid of coordinates
        ...     y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        ...     
        ...     # Calculate the distance from the center of the image
        ...     center_x, center_y = w // 2, h // 2
        ...     dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        ...     
        ...     # Calculate the angle from the center of the image
        ...     angle_from_center = torch.atan2(y - center_y, x - center_x)
        ...     
        ...     # Calculate the wave pattern based on the distance and angle
        ...     wave_freq = 0.05  # Frequency of the waves
        ...     wave_amp = 10.0   # Amplitude of the waves
        ...     wave_offset = frame * 0.05  # Offset for animation
        ...     
        ...     dx = wave_amp * torch.cos(dist_from_center * wave_freq + angle_from_center + wave_offset)
        ...     dy = wave_amp * torch.sin(dist_from_center * wave_freq + angle_from_center + wave_offset)
        ...     
        ...     return dx, dy
        ... 
        >>> def demo_wiggly_dog():
        ...     real_image = as_torch_image(
        ...         rp.cv_resize_image(
        ...             load_image(
        ...                 "https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg"
        ...             ),
        ...             (512, 512),
        ...         )
        ...     )
        ...     
        ...     #real_image = torch.rand(real_image.shape)
        ...     
        ...     out_frames = []
        ...     
        ...     for frame in range(200):
        ...         h, w = get_image_dimensions(real_image)
        ...         
        ...         dx, dy = calculate_wave_pattern(h, w, frame)
        ...         
        ...         warped_image = torch_remap_image(real_image, dx, dy, relative=True, interp='bilinear', add_alpha_mask=True)    
        ...         warped_image = as_numpy_image(warped_image)
        ...         warped_image = with_alpha_checkerboard(warped_image)
        ...         out_frames.append(warped_image)
        ...         
        ...     display_video(out_frames)
        ... 
        >>> demo_wiggly_dog()

    Note: For interp='nearest', will x and y be floored or rounded?
        Answer: The values will be rounded! Shifting x relative by -.01 will not make any change!
        EXAMPLE ANALYSIS:
            >>> load_image('https://avatars.githubusercontent.com/u/17944835?v=4&size=64')
            >>> image = as_torch_image(load_image('https://avatars.githubusercontent.com/u/17944835?v=4&size=64'))
            >>> x, y = image[:2] * 0

            >>> #Testing what happens when we shift with nearest interp...it rounds the positions!
            >>> display_image(torch_remap_image(image, x    , y, relative=True                  )) # Does not change the image
            >>> display_image(torch_remap_image(image, x+1  , y, relative=True                  )) # Shifts image to left by one pixel  
            >>> display_image(torch_remap_image(image, x-1  , y, relative=True                  )) # Shifts image to right by one pixel
            >>> display_image(torch_remap_image(image, x    , y, relative=True, interp='nearest')) # Same as with bilinear - does nothing
            >>> display_image(torch_remap_image(image, x+.01, y, relative=True, interp='nearest')) # Does not change anything!
            >>> display_image(torch_remap_image(image, x-.01, y, relative=True, interp='nearest')) # Does not change anything!
            >>> display_image(torch_remap_image(image, x-.49, y, relative=True, interp='nearest')) # Does not change anything!  
            >>> display_image(torch_remap_image(image, x-.5 , y, relative=True, interp='nearest')) # Changes stuff. Some pixels are shifted, some aren't
            >>> display_image(torch_remap_image(image, x-.51, y, relative=True, interp='nearest')) # Just like x-1, shifts image to right by 1 pixel

            >>> #Checking 'nearest' recursive stability - it passes!
            >>> new_image=image
            >>> for _ in range(10):
            ...     display_image(new_image)
            ...     new_image = torch_remap_image(new_image, x, y, relative=True, interp="nearest")
            >>> assert (image==new_image).all()

            >>> #Checking 'bilinear' recursive stability - it numerically  passes, but is not bitwise-perfect
            >>> new_image=image
            >>> for _ in range(10):
            ...     display_image(new_image)
            ...     new_image = torch_remap_image(new_image, x, y, relative=True, interp="bilinear")
            >>> print((image==new_image).all()     ) #Printed: tensor(False)
            >>> print((image-new_image).abs().max()) #Printed: tensor(8.3447e-06)
    """

    import torch
    import torch.nn.functional as F
    from einops import rearrange

    in_c, in_height, in_width = image.shape
    out_height, out_width = x.shape

    if add_alpha_mask:
        alpha_mask = torch.ones_like(image[:1])
        image = torch.cat([image, alpha_mask], dim=0)

    if relative:
        assert in_height == out_height, "For relative warping, input and output heights must match, but got in_height={} and out_height={}".format(in_height, out_height)
        assert in_width  == out_width , "For relative warping, input and output widths must match, but got in_width={} and out_width={}".format(in_width, out_width)
        x = x + torch.arange(in_width , device=x.device, dtype=x.dtype)
        y = y + torch.arange(in_height, device=y.device, dtype=y.dtype)[:,None]

    x = rearrange(x, "h w -> 1 h w")
    y = rearrange(y, "h w -> 1 h w")

    # Normalize coordinates to [-1, 1] range - which F.grid_sample requires
    x = (x / (in_width - 1)) * 2 - 1
    y = (y / (in_height - 1)) * 2 - 1

    # Stack x and y coordinates
    grid = torch.stack([x, y], dim=-1)
    grid = grid.to(image.dtype)

    # Choose an interpolation method
    interp_methods = {
        "bilinear": "bilinear",
        "bicubic": "bicubic",
        "nearest": "nearest",
    }
    assert interp in interp_methods, 'torch_remap_image: interp must be one of the following: {}'.format(list(interp_methods))
    interp_mode = interp_methods[interp]

    # Remap the image using grid_sample
    out = F.grid_sample(rearrange(image, "c h w -> 1 c h w"), grid, mode=interp_mode, align_corners=True)
    out = rearrange(out, "1 c h w -> c h w")

    # Assert the shape of the output tensor
    expected_c = in_c+1 if add_alpha_mask else in_c
    assert out.shape == (expected_c, out_height, out_width), "Expected output shape: ({}, {}, {}), but got: {}".format(expected_c, out_height, out_width, out.shape)

    return out

import math
def _ceil(x):
    """ Works across libraries - such as numpy, torch, pure python """
    if isinstance(x, np.ndarray):return np.ceil(x).astype(int)
    if isinstance(x, torch.Tensor):return x.ceil().long()
    return math.ceil(x)

def _floor(x):
    """ Works across libraries - such as numpy, torch, pure python """
    if isinstance(x, np.ndarray):return np.floor(x).astype(int)
    if isinstance(x, torch.Tensor):return x.floor().long()
    return math.floor(x)

def get_bilinear_weights(x, y):
    """
    Calculate bilinear interpolation weights for a set of (x, y) coordinates.

    This function takes a set of (x, y) coordinates and returns the corresponding
    bilinear interpolation weights and the integer coordinates of the 4 neighboring
    pixels for each input coordinate.

    The math behind this function is explained here:
        https://www.desmos.com/calculator/esool5qrrd

    Args:
        x (torch.Tensor or numpy.ndarray): The x-coordinates of the input points.
        y (torch.Tensor or numpy.ndarray): The y-coordinates of the input points.

    Returns:
        tuple: A tuple containing three elements:
            - X (torch.Tensor or numpy.ndarray): The integer x-coordinates of the 4 neighboring pixels for each input point. Shape: (4, *x.shape)
            - Y (torch.Tensor or numpy.ndarray): The integer y-coordinates of the 4 neighboring pixels for each input point. Shape: (4, *y.shape)
            - W (torch.Tensor or numpy.ndarray): The bilinear interpolation weights for each of the 4 neighboring pixels. Shape: (4, *x.shape)

    Note:
        - x and y should have the same shape.
        - This function works with both PyTorch tensors and NumPy arrays.
        - Was originally called calculate_subpixel_weights, featured in TRITON's source/unprojector.py 
            (see https://github.com/TritonPaper/TRITON/blob/master/source/unprojector.py)
    """
    assert x.shape == y.shape, "x and y must have the same shape"
    shape = x.shape

    Rx = x % 1
    Ry = y % 1
    Qx = 1 - Rx
    Qy = 1 - Ry

    A=Rx*Ry #Weight for  ceil(x), ceil(y)
    B=Rx*Qy #Weight for  ceil(x),floor(y)
    C=Qx*Qy #Weight for floor(x),floor(x)
    D=Qx*Ry #Weight for floor(x), ceil(y)

    Cx = _ceil(x)
    Cy = _ceil(y)
    Fx = _floor(x)
    Fy = _floor(y)

    stack = torch.stack if isinstance(x, torch.Tensor) else np.stack
    
    X = stack((Cx, Cx, Fx, Fx))  # All X values
    Y = stack((Cy, Fy, Fy, Cy))  # All Y values
    W = stack((A, B, C, D))      # Weights

    assert X.shape == (4, *shape), "Expected X.shape == (4, *x.shape), but got {}".format(X.shape)
    assert Y.shape == (4, *shape), "Expected Y.shape == (4, *y.shape), but got {}".format(Y.shape)
    assert W.shape == (4, *shape), "Expected W.shape == (4, *x.shape), but got {}".format(W.shape)

    return X, Y, W

def torch_scatter_add_image(image, x, y, *, relative=False, interp='floor', height=None, width=None, prepend_ones=False):
    """
    Scatter add an image tensor using the given x and y coordinate tensors.
    Pixels warped out-of-bounds will be skipped.
    This is similar to torch_remap_image, but uses scatter_add instead of remapping.
    
    If relative=True, it will treat x and y as deltas (dx and dy) and perform relative scatter adding.
        Note: Because this is a scatter operation, the direction of movement will be the same as dx and dy.

    Args:
        image (torch.Tensor): Input image tensor of shape [C, H, W], where C is the number of channels (e.g., 3 for RGB, 4 for RGBA),
                              H is the height, and W is the width of the image.
        x (torch.Tensor): X-coordinate tensor of shape [H_in, W_in] specifying the x-coordinates for scatter adding,
                          where H_in is the input height and W_in is the input width.
        y (torch.Tensor): Y-coordinate tensor of shape [H_in, W_in] specifying the y-coordinates for scatter adding.
        relative (bool, optional): If True, treat x and y as deltas (dx and dy) and perform relative scatter adding. Default is False.
        interp (str, optional): Specifies how to handle fractional coordinates. Can be one of 'bilinear' (the slowest one), 'floor' (the fastest one), 'ceil', or 'round'. Default is 'floor'.
        height (int, optional): The output height. If not specified, it is inferred from the input image height.
        width (int, optional): The output width. If not specified, it is inferred from the input image width.
        prepend_ones (bool, optional): If True, prepends a channel of ones to the input tensor before calculation. Useful for getting the sum easily by accessing output[0]. This option is simply for convenience. Default is False.

    Returns:
        torch.Tensor: Scatter added image tensor of shape [C, H_out, W_out] or [C+1, H_out, W_out] if prepend_ones is True, where:
            - C is the number of channels in the input image.
            - H_out is the output height.
            - W_out is the output width.

    EXAMPLE (image warping):

        >>> def demo_torch_scatter_add_image(interp,normalize=False,):
        ...     import torch
        ...
        ...     url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        ...     image=as_torch_image(load_image(url))
        ...     
        ...     # Define the animation parameters
        ...     num_frames = 100
        ...     wave_speed = 0.1
        ...     wave_freq = 0.025
        ...     wave_amp = 20
        ...
        ...     def get_frames():
        ...         for frame in range(num_frames):
        ...             y, x = torch.meshgrid(torch.arange(image.shape[1]), torch.arange(image.shape[2]))
        ...             x = x.float()
        ...             y = y.float()
        ...
        ...             # Apply sine and cosine waves to create caustics effect
        ...             dx = wave_amp * torch.cos(wave_freq * (x + y) + frame * wave_speed)
        ...             dy = wave_amp * torch.sin(wave_freq * (x - y) + frame * wave_speed)
        ...
        ...             caustics_image = torch_scatter_add_image(
        ...                 image,
        ...                 dx,
        ...                 dy,
        ...                 relative=True,
        ...                 interp=interp,
        ...                 prepend_ones=True
        ...             )
        ...             total, caustics_image = caustics_image[0],caustics_image[1:]
        ...             if normalize:
        ...                 caustics_image/=total
        ...
        ...             # Append the current frame to the list of frames
        ...             yield caustics_image
        ...
        ...     display_video(get_frames(),framerate=60)
        ...
        ... demo_torch_scatter_add_image('floor')
        ... demo_torch_scatter_add_image('floor',normalize=True)
        ... demo_torch_scatter_add_image('bilinear',normalize=True)

    EXAMPLE (image resizing):

        >>> def demo_torch_scatter_add_image_resizing(interp,normalize=False,):
        ...     import torch
        ... 
        ...     url = "https://content.whrb.org/files/3916/1997/2511/Lindsey_Stirling_2021_by_F._Scott_Schafer.png"
        ...     image=load_image(url)
        ...     image=resize_image_to_fit(image,height=256,width=256)
        ...     image=as_torch_image(image)
        ...     num_frames=300
        ...     
        ...     old_height, old_width = get_image_dimensions(image)
        ...     
        ...     def get_frames():
        ...         for frame in range(num_frames):
        ...             new_width =3*frame+10
        ...             new_height=3*frame+10
        ...         
        ...             x, y = xy_torch_matrices(
        ...                 old_height,
        ...                 old_width,
        ...                 max_x=new_width,
        ...                 max_y=new_height,
        ...             )
        ...             
        ...             resized_image = torch_scatter_add_image(
        ...                 image,
        ...                 x,
        ...                 y,
        ...                 height=new_height,
        ...                 width=new_width,
        ...                 interp=interp,
        ...                 prepend_ones=True,
        ...             )
        ...             
        ...             total, resized_image = resized_image[0],resized_image[1:]
        ...             
        ...             if normalize:
        ...                 resized_image/=total
        ... 
        ...             #Create the preview image
        ...             resized_image=as_numpy_image(resized_image)
        ...             resized_image=bordered_image_solid_color(resized_image,color='red')
        ...             resized_image=crop_image(resized_image,height=500,width=500)
        ...             resized_image = labeled_image(
        ...                 resized_image,
        ...                 f"{interp} normalize={normalize} {(new_width,new_height)}",
        ...                 font="G:Quicksand",
        ...             )
        ... 
        ...             # Append the current frame to the list of frames
        ...             yield resized_image
        ... 
        ...     display_video(get_frames(),framerate=60)
        ... 
        ... demo_torch_scatter_add_image_resizing('bilinear',normalize=True)
        ... demo_torch_scatter_add_image_resizing('floor',normalize=False)
        ... demo_torch_scatter_add_image_resizing('floor',normalize=True)
        ... demo_torch_scatter_add_image_resizing('bilinear',normalize=False)

    EXAMPLE (noise warping):

        >>> def resize_noise(noise, new_height, new_width):
        ...     #I'm putting this in rp.git_import('CommonSource').noise_warp
        ...     #Can resize gaussian noise, adjusting for variance and preventing cross-correlation
        ...     
        ...     C, old_height, old_width = noise.shape
        ... 
        ...     x, y = xy_torch_matrices(
        ...         old_height,
        ...         old_width,
        ...         max_x=new_width,
        ...         max_y=new_height,
        ...     )
        ... 
        ...     resized = torch_scatter_add_image(
        ...         noise,
        ...         x,
        ...         y,
        ...         height=new_height,
        ...         width=new_width,
        ...         interp='round',
        ...         #interp='bilinear', #This introduces cross correlation! Can't use this for noise warping.
        ...         prepend_ones=True
        ...     )
        ... 
        ...     total, resized = resized[:1], resized[1:]
        ... 
        ...     adjusted = resized / total**.5
        ... 
        ...     return adjusted
        ... 
        ... import torch
        ... frames=[]
        ... base_noise=torch.randn(3,512,512)
        ... for size in resize_list(range(10,1024),512):
        ...     new_noise=resize_noise(base_noise,size,size)
        ...     
        ...     frame=as_numpy_image(new_noise)/5+.5
        ...     frame=bordered_image_solid_color(frame,'red')
        ...     frame=crop_image(frame,height=1024,width=1024)
        ...     frame=labeled_image(frame,f"size={size}, mean={float(new_noise.mean()):.2} std={float(new_noise.std()):.2}")
        ...     frames.append(frame)
        ...     display_image(frame)
        ...     
        ... ans=printed(save_video_mp4(frames,video_bitrate='max'))
        ... display_video(frames,loop=True)


    """

    import torch
    from einops import rearrange

    if prepend_ones:
        #We might prepend a channel of ones to keep track of how many were added so we can normalize later
        #For example, to get the mean we would divide output[1:]/output[0]
        ones = torch.ones_like(image[:1])
        image = torch.cat([ones, image], dim=0)

    in_c, in_height, in_width = image.shape
    out_height, out_width = x.shape

    assert y.shape == x.shape == (in_height, in_width), "x and y should have the same height and width as the input image, aka {} but x.shape=={} and y.shape=={}".format((in_height, in_width),x.shape,y.shape)
    
    # If we don't specify the output width and height in args, copy the height and width of the input image
    out_width = width if width is not None else in_width
    out_height = height if height is not None else in_height
    
    x = x.to(image.device)
    y = y.to(image.device)

    # Apply the specified rounding interp to the coordinates
    if interp == 'bilinear':
        #A form of antialiasing
        components = []
        for X, Y, W in zip(*get_bilinear_weights(x, y)):
            components.append(
                torch_scatter_add_image(
                    image * W[None],
                    X,
                    Y,
                    relative=relative,
                    interp="floor",
                    height=height,
                    width=width,
                    prepend_ones=False,
                )
            )
        return sum(components)
    if interp == 'round':
        x = x.round()
        y = y.round()
    elif interp == 'ceil':
        x = x.ceil()
        y = y.ceil()
    else:
        assert interp == 'floor'
        # Will be implicitly floored during conversion

    # Make sure x and y are int64 values, for indexing in torch_scatter
    x = x.long()
    y = y.long()

    if relative:
        assert in_height == out_height, "For relative scatter adding, input and output heights must match, but got in_height={} and out_height={}".format(in_height, out_height)
        assert in_width == out_width, "For relative scatter adding, input and output widths must match, but got in_width={} and out_width={}".format(in_width, out_width)
        x = x + torch.arange(in_width , device=x.device, dtype=x.dtype)
        y = y + torch.arange(in_height, device=y.device, dtype=y.dtype)[:,None]

    # Initialize the output tensor with zeros
    out = torch.zeros((out_height * out_width, in_c), dtype=image.dtype, device=image.device)

    # Compute the flattened indices for scatter_add
    # And Filter out out-of-bounds indices based on the specified output height and width
    indices = y * out_width + x
    valid_indices = (y >= 0) & (y < out_height) & (x >= 0) & (x < out_width)
    valid_mask = rearrange(valid_indices, "h w -> (h w)")
    indices    = rearrange(indices, "h w -> (h w)")
    valid_indices = indices[valid_mask]

    # Flatten the image tensor
    flat_image = rearrange(image, "c h w -> (h w) c")
    valid_flat_image = flat_image[valid_mask]

    # Perform scatter_add operation using torch.index_add
    out.index_add_(0, valid_indices, valid_flat_image)

    # Reshape the output tensor back to the original shape
    out = rearrange(out, "(h w) c -> c h w", h=out_height, w=out_width)

    return out