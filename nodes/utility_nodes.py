import math
import torch
import torchvision
from .. import RyanOnTheInside
from abc import ABC
from ..tooltips import apply_tooltips
 
class UtilityNode(RyanOnTheInside, ABC):
    #NOTE: for forward compatibility
    CATEGORY="RyanOnTheInside/Utility"

class BatchUtilityNode(UtilityNode):
    #NOTE: for forward compatibility
    CATEGORY= f"{UtilityNode.CATEGORY}/Batches"

@apply_tooltips
class ImageIntervalSelect(BatchUtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "interval": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "end_at": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }
    
    def select_interval(self, image, interval=1, start_at=0, end_at=0):
        # Set default for end_at if it is None
        if end_at == 0:
            end_at = len(image)
        
        # Ensure start_at and end_at are within the bounds of the image list
        start_at = max(0, min(start_at, len(image) - 1))
        end_at = max(start_at + 1, min(end_at, len(image)))

        # Slice the image list from start_at to end_at with the specified interval
        images = image[start_at:end_at:interval]
        return (images,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_interval"

@apply_tooltips
class ImageIndexSelect(BatchUtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "indices": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "forceInput": True}),
                "filter_behavior": (["ignore", "error", "wrap"], {"default": "wrap"}),
            },
        }
    
    def select_indices(self, image, indices, filter_behavior="ignore"):
        # Convert single index to list if necessary
        if not isinstance(indices, list):
            indices = [indices]
        
        batch_size = len(image)
        
        if filter_behavior == "error":
            # Check if any index is out of bounds
            invalid_indices = [idx for idx in indices if idx < 0 or idx >= batch_size]
            if invalid_indices:
                raise ValueError(f"Indices {invalid_indices} are out of bounds for batch size {batch_size}")
            valid_indices = indices
        elif filter_behavior == "wrap":
            # Use modulo to wrap indices around
            valid_indices = [idx % batch_size for idx in indices]
        else:  # "ignore"
            # Filter out invalid indices
            valid_indices = [idx for idx in indices if 0 <= idx < batch_size]
        
        if not valid_indices:
            raise ValueError("No valid indices provided")
        
        # Select the images at the specified indices
        selected_images = image[valid_indices]
        return (selected_images,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_indices"

#NOTE eh FIX MEH
@apply_tooltips
class ImageIntervalSelectPercentage(BatchUtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "interval_percentage": ("FLOAT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "start_percentage": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "end_percentage": ("FLOAT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
        }
    
    def select_percentage_interval(self, image, interval_percentage=10, start_percentage=0, end_percentage=100):
        total_images = len(image)
        interval = max(1, int(total_images * (interval_percentage / 100)))
        start_at = int(total_images * (start_percentage / 100))
        end_at = int(total_images * (end_percentage / 100))

        # Ensure start_at and end_at are within the bounds of the image list
        start_at = max(0, min(start_at, total_images - 1))
        end_at = max(start_at + 1, min(end_at, total_images))

        # Slice the image list from start_at to end_at with the specified interval
        images = image[start_at:end_at:interval]
        return (images,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_percentage_interval"

@apply_tooltips
class ImageChunks(BatchUtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            },
        }
    
    def concatenate_images_into_grid(self, image, padding=0, normalize=False, scale_each=False, pad_value=0):
        image = image.permute(0, 3, 1, 2)
        num_images = image.shape[0]
        grid_size = math.ceil(math.sqrt(num_images))
        nrow = grid_size

        if num_images < grid_size ** 2:
            num_to_add = grid_size ** 2 - num_images
            black_image = torch.zeros((num_to_add, image.shape[1], image.shape[2], image.shape[3]), dtype=image.dtype, device=image.device)
            image = torch.cat((image, black_image), dim=0)

        grid = torchvision.utils.make_grid(image, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value)
        grid = grid.unsqueeze(0).permute(0, 2, 3, 1)
        grid = grid.to('cpu').to(torch.float32)
        return (grid,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate_images_into_grid"

#TODO inherit from ImageChunk reuse
@apply_tooltips
class VideoChunks(BatchUtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "chunk_size": ("INT", {"default": 4, "min": 1})
            },
        }
    
    def chunk_images_into_grids(self, image, chunk_size=4, padding=2, normalize=False, scale_each=False, pad_value=0):
        # Ensure the input tensor is in the correct format for torchvision.utils.make_grid
        image = image.permute(0, 3, 1, 2)  # Convert from NHWC to NCHW if necessary

        grids = []
        for i in range(0, len(image), chunk_size):
            # Extract a chunk of images
            chunk = image[i:i+chunk_size]

            # If the chunk is smaller than chunk_size, append black images to make up the difference
            if len(chunk) < chunk_size:
                num_black_images = chunk_size - len(chunk)
                black_images = torch.zeros((num_black_images, *chunk.shape[1:]), dtype=chunk.dtype, device=chunk.device)
                chunk = torch.cat((chunk, black_images), dim=0)

            nrow = math.ceil(math.sqrt(chunk_size))
            grid = torchvision.utils.make_grid(chunk, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value)
            #grid = grid.unsqueeze(0).permute(0, 2, 3, 1)
            #grid = grid.to('cpu').to(torch.float32)
            grids.append(grid)

        # Combine all grids into a single tensor
        # Assuming grids are of the same size, we can use torch.stack
        combined_grids = torch.stack(grids, dim=0)

        # Convert back to NHWC for consistency if needed
        combined_grids = combined_grids.permute(0, 2, 3, 1).to('cpu').to(torch.float32)

        return (combined_grids,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "chunk_images_into_grids"

@apply_tooltips
class Image_Shuffle(BatchUtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shuffle_size": ("INT", {"default": 4, "min": 1})
            },
        }

    def shuffle_images(self, image, shuffle_size=4):
        # Ensure shuffle_size is within the bounds of the image batch
        shuffle_size = min(shuffle_size, image.shape[0])

        # Shuffle the images in groups of shuffle_size
        shuffled_images = []
        for i in range(0, len(image), shuffle_size):
            chunk = image[i:i+shuffle_size]
            indices = torch.randperm(chunk.shape[0])
            shuffled_chunk = chunk[indices]
            shuffled_images.append(shuffled_chunk)

        # Concatenate all shuffled chunks back into a single tensor
        shuffled_images = torch.cat(shuffled_images, dim=0)

        return (shuffled_images,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shuffle_images"

@apply_tooltips
class ImageDifference(UtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    def compute_difference(self, image):
        # Ensure there are at least two images to compute the difference
        if image.shape[0] < 2:
            raise ValueError("The batch must contain at least two images to compute the difference.")

        # Compute the difference between each image and the previous image
        differences = image[1:] - image[:-1]

        # Add a zero tensor at the beginning to maintain the same batch size
        zero_tensor = torch.zeros_like(image[0:1])
        differences = torch.cat((zero_tensor, differences), dim=0)

        return (differences,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute_difference"

@apply_tooltips
class SwapDevice(UtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cpu", "cuda"],),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    def swap_device(self, device, image=None, mask=None):
        # Check if the device is available
        if device not in ["cpu", "cuda"] or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError(f"Device {device} is not available.")

        # Transfer image to the chosen device or create a zero tensor if image is None
        if image is not None:
            image = image.to(device)
        else:
            image = torch.zeros((1, 1, 1, 1), device=device)

        # Transfer mask to the chosen device or create a zero tensor if mask is None
        if mask is not None:
            mask = mask.to(device)
        else:
            mask = torch.zeros((1, 1, 1, 1), device=device)

        return (image, mask)

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "swap_device"