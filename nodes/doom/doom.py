import os
import folder_paths
from server import PromptServer
import numpy as np
import cv2
import torch

class Doom_:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            }
        }   

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_doom"
    CATEGORY = "DOOM"

    def run_doom(self):
       img = np.zeros((512, 768, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       
       # Draw header
       header = "RyanOnTheInside"
       header_size = cv2.getTextSize(header, font, 2, 3)[0]
       headerX = (img.shape[1] - header_size[0]) // 2
       headerY = img.shape[0] // 4
       cv2.putText(img, header, (headerX, headerY), font, 2, (255,255,255), 3)

       # Draw social links
       links = [
           "@RyanOnTheInside everywhere",
           "Node Suite:",
           "github.com/ryanontheinside/ComfyUI_RyanOnTheInside"
       ]
       
       y_offset = img.shape[0] // 2
       for text in links:
           textsize = cv2.getTextSize(text, font, 0.7, 2)[0]
           textX = (img.shape[1] - textsize[0]) // 2
           cv2.putText(img, text, (textX, y_offset), font, 0.7, (255,255,255), 2)
           y_offset += 50

       tensor = torch.from_numpy(img).float() / 255.0
       tensor = tensor.unsqueeze(0)
       return (tensor,)
      

