from .features import WhisperFeature, BaseFeature
from .feature_extractors import FeaturePipe, FirstFeature
import json
from typing import Dict, List, Tuple, Union
import torch
import torch.nn.functional as F
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont
import numpy as np

#NOTE: below is an example of the data we are expecting. 
# This is built to work with ComfyUI-Whisper, but only the json structure below is required.
#   # text:
    # We're telling a story about water and then about fire and then we're going to do ice and then psychedelic awesomeness. Okay.

    # #segement_alignment
    # [{"value": "We're telling a story about water and then about fire and then we're going to do ice and then", "start": 0.0, "end": 6.26}, {"value": "psychedelic awesomeness. Okay.", "start": 7.9799999999999995, "end": 9.38}]

    # #words_alignment
    # [{"value": "We're", "start": 0.0, "end": 0.26}, {"value": "telling", "start": 0.26, "end": 0.46}, {"value": "a", "start": 0.46, "end": 0.6}, {"value": "story", "start": 0.6, "end": 0.88}, {"value": "about", "start": 0.88, "end": 1.14}, {"value": "water", "start": 1.14, "end": 1.58}, {"value": "and", "start": 1.58, "end": 2.24}, {"value": "then", "start": 2.24, "end": 2.44}, {"value": "about", "start": 2.44, "end": 3.24}, {"value": "fire", "start": 3.24, "end": 3.66}, {"value": "and", "start": 3.66, "end": 4.1}, {"value": "then", "start": 4.1, "end": 4.3}, {"value": "we're", "start": 4.3, "end": 4.7}, {"value": "going", "start": 4.7, "end": 4.88}, {"value": "to", "start": 4.88, "end": 5.02}, {"value": "do", "start": 5.02, "end": 5.18}, {"value": "ice", "start": 5.18, "end": 5.44}, {"value": "and", "start": 5.44, "end": 5.98}, {"value": "then", "start": 5.98, "end": 6.26}, {"value": "psychedelic", "start": 7.9799999999999995, "end": 8.52}, {"value": "awesomeness.", "start": 8.52, "end": 9.06}, {"value": "Okay.", "start": 9.2, "end": 9.38}]

class WhisperFeatureNode(FirstFeature):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return WhisperFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "alignment_data": ("STRING", {"multiline": True}),
            },
            "optional": {
                "trigger_set": ("TRIGGER_SET",),
                "context_size": ("INT", {"default": +3, "min": 0, "max": 10}),
                "overlap_mode": (["blend", "replace", "add"], {"default": "blend"}),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "create_feature"

    def create_feature(self, video_frames, frame_rate, extraction_method, 
                      alignment_data, trigger_set=None, context_size=3, 
                      overlap_mode="blend"):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        
        # Parse trigger set if provided
        triggers = None
        if trigger_set:
            try:
                triggers = json.loads(trigger_set)
            except json.JSONDecodeError:
                print("Warning: Could not parse trigger_set JSON")
        
        # Create the WhisperFeature without context_options
        whisper_feature = WhisperFeature(
            name="whisper_feature",
            frame_rate=feature_pipe.frame_rate,
            frame_count=len(video_frames),
            alignment_data=alignment_data,
            trigger_pairs=triggers,
            feature_name=extraction_method
        )
        
        whisper_feature.extract()
        return (whisper_feature, feature_pipe)

class TriggerBuilder:
    """Creates triggers that respond to specific words or phrases in the Whisper transcription.
    
    Example Usage:
        1. To highlight when someone says "amazing":
           - pattern: "amazing"
           - start_value: 0.0
           - end_value: 1.0
           - match_mode: "contains"
           - fade_type: "linear"
           
        2. To fade out during questions:
           - pattern: "\\?$"  # Regex for question mark at end
           - start_value: 1.0
           - end_value: 0.0
           - match_mode: "regex"
           - fade_type: "smooth"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # The word or phrase to match in the transcription
                "pattern": ("STRING", {"multiline": False, "default": "hello"}),
                
                # Value when the word starts (0-1)
                "start_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                
                # Value when the word ends (0-1)
                "end_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                
                # How to match the pattern:
                # - exact: Must match whole word exactly
                # - contains: Pattern appears anywhere in word
                # - regex: Use regular expressions
                # - phonetic: Match similar sounding words
                "match_mode": (["exact", "contains", "regex", "phonetic"], {"default": "contains"}),
                
                # How to transition between values:
                # - none: Instant change
                # - linear: Smooth linear fade
                # - smooth: Smooth eased transition
                "fade_type": (["none", "linear", "smooth"], {"default": "linear"}),
                
                # Duration in frames (0 = use word duration)
                "duration_frames": ("INT", {"default": 0, "min": 0}),
            }
        }
    
    RETURN_TYPES = ("TRIGGER",)
    FUNCTION = "build"
    CATEGORY = "RyanOnTheInside/FlexFeatures/Triggers"

    def build(self, pattern: str, start_value: float, end_value: float, 
             match_mode: str, fade_type: str, duration_frames: int) -> tuple:
        trigger = {
            "pattern": pattern,
            "values": (start_value, end_value),
            "mode": match_mode,
            "fade": fade_type,
            "duration": duration_frames
        }
        return (json.dumps(trigger),)

class TriggerCombiner:
    """Combines multiple triggers into a single trigger set with blending.
    
    Example Usage:
        1. Blend two complementary triggers:
           - trigger1: Highlight "happy" (0->1)
           - trigger2: Highlight "sad" (1->0)
           - blend_mode: "blend"
           
        2. Add multiple intensity triggers:
           - trigger1: Response to "very"
           - trigger2: Response to "extremely"
           - blend_mode: "add"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # How to combine multiple triggers:
                # - blend: Smooth interpolation between triggers
                # - add: Sum all trigger values (clamp to 1.0)
                # - multiply: Multiply trigger values together
                # - max: Use highest trigger value
                "blend_mode": (["blend", "add", "multiply", "max"], {"default": "blend"}),
            },
            "optional": {
                # Connect multiple TriggerBuilder outputs here
                "trigger1": ("TRIGGER",),
                "trigger2": ("TRIGGER",),
                "trigger3": ("TRIGGER",),
                "trigger4": ("TRIGGER",),
            }
        }
    
    RETURN_TYPES = ("TRIGGER_SET",)
    FUNCTION = "combine"
    CATEGORY = "RyanOnTheInside/FlexFeatures/Triggers"

    def combine(self, blend_mode: str, trigger1=None, trigger2=None, 
                trigger3=None, trigger4=None) -> tuple:
        triggers = []
        for t in [trigger1, trigger2, trigger3, trigger4]:
            if t is not None:
                triggers.append(json.loads(t))
        
        trigger_set = {
            "triggers": triggers,
            "blend_mode": blend_mode
        }
        return (json.dumps(trigger_set),)

class ContextModifier:
    """Modifies trigger behavior based on context.
    
    Example Usage:
        1. Amplify long words:
           - modifier_type: "timing"
           - condition: "duration > 0.5"
           - value_adjust: 1.5
           
        2. Create sequence effects:
           - modifier_type: "sequence"
           - condition: "index % 2 == 0"
           - value_adjust: 0.8
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_set": ("TRIGGER_SET",),
                
                # Type of context to consider:
                # - timing: Word duration and position
                # - sentiment: Positive/negative context
                # - speaker: Who is speaking
                # - sequence: Pattern in word sequence
                "modifier_type": (["timing", "sentiment", "speaker", "sequence"], 
                                {"default": "timing"}),
                
                # Python expression that determines when to apply modification
                # Available variables depend on modifier_type:
                # - timing: duration, start, end
                # - sentiment: is_positive, sentiment_score
                # - speaker: speaker_id, is_new_speaker
                # - sequence: index, total_words
                "condition": ("STRING", {"multiline": True, 
                                       "default": "duration > 0.5"}),
                
                # How much to modify the trigger value when condition is true
                # 1.0 = no change, >1.0 amplify, <1.0 reduce
                "value_adjust": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0}),
                
                # How many words to look at for context
                "window_size": ("INT", {"default": 3, "min": 1, "max": 10}),
            }
        }
    
    RETURN_TYPES = ("TRIGGER_SET",)
    FUNCTION = "modify"
    CATEGORY = "RyanOnTheInside/FlexFeatures/Triggers"

    def modify(self, trigger_set: str, modifier_type: str, condition: str, 
               value_adjust: float, window_size: int) -> tuple:
        trigger_data = json.loads(trigger_set)
        
        context_mod = {
            "type": modifier_type,
            "condition": condition,
            "adjustment": value_adjust,
            "window": window_size
        }
        
        trigger_data["context_modifier"] = context_mod
        return (json.dumps(trigger_data),)

import json
class WhisperToPromptTravel:
    """Converts Whisper alignment data to prompt travel format"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments_alignment": ("STRING", {"multiline": True}), # JSON string of segment alignments
                "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 120.0}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "misc"

    def convert(self, segments_alignment, fps):
        import json
        
        # Parse the alignment data
        try:
            segments = json.loads(segments_alignment)
        except json.JSONDecodeError:
            raise ValueError("Invalid segments_alignment JSON format")
        
        # Create frame-to-prompt mapping
        prompt_travel = {}
        
        for segment in segments:
            # Convert time to frame number
            frame = int(segment["start"] * fps)
            prompt_travel[str(frame)] = segment["value"]
        
        # Convert to string format
        result = "{\n"
        for frame, prompt in sorted(prompt_travel.items(), key=lambda x: int(x[0])):
            result += f'"{frame}":"{prompt}",\n'
        result = result.rstrip(",\n") + "\n}"
        
        return (result,)

class WhisperTextRenderer:
    """Renders text overlays from Whisper alignment data with animations"""
    
    # Built-in fonts that are OS-agnostic
    BUILTIN_FONTS = {
        "arial": "arial.ttf",
        "times": "times.ttf",
        "courier": "courier.ttf",
        "helvetica": "helvetica.ttf"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Input video frames
                "feature": ("FEATURE",),
                "font_size": ("INT", {"default": 32, "min": 8, "max": 256}),
                "font_name": (list(cls.BUILTIN_FONTS.keys()), {"default": "arial"}),
                "position": (["top", "middle", "bottom"], {"default": "bottom"}),
                "horizontal_align": (["left", "center", "right"], {"default": "center"}),
                "margin": ("INT", {"default": 20, "min": 0, "max": 200}),
                "animation_type": (["none", "fade", "pop", "slide"], {"default": "fade"}),
                "animation_duration": ("INT", {"default": 15, "min": 1, "max": 60}),
            },
            "optional": {
                "max_width": ("INT", {"default": 0, "min": 0}),  # 0 = full width
                "bg_color": ("STRING", {"default": "#000000"}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "RyanOnTheInside/FlexFeatures/Text"

    def render(self, images, feature, font_size, font_name, position, horizontal_align,
               margin, animation_type, animation_duration, max_width=0,
               bg_color="#000000", text_color="#FFFFFF", opacity=0.8):
        
        # Get dimensions from input images
        B, H, W, C = images.shape
        device = images.device
        
        # Create output tensor
        output = torch.zeros((B, H, W, 4), device=device)
        output[..., 3] = 1.0  # Set alpha to 1
        
        # Load font
        font_path = self._get_font_path(font_name)
        font = ImageFont.truetype(str(font_path), font_size)
        
        # Process each word/segment
        for item in feature.alignment_data:
            text = item["value"]
            start_frame = int(item["start"] * feature.frame_rate)
            end_frame = int(item["end"] * feature.frame_rate)
            
            # Create text mask using PIL (temporarily on CPU)
            text_mask = self._create_text_mask(
                text, (W, H), font, position, horizontal_align, 
                margin, max_width
            )
            
            # Convert mask to tensor and move to GPU
            text_mask = torch.from_numpy(text_mask).to(device)
            
            # Apply animation and colors
            self._apply_animation_and_colors(
                output, text_mask, start_frame, end_frame,
                animation_type, animation_duration,
                text_color, bg_color, opacity
            )
        
        # Composite with input images
        output = images * (1 - output[..., 3:]) + output[..., :3] * output[..., 3:]
        
        return (output,)

    def _create_text_mask(self, text, size, font, position, align, margin, max_width):
        """Creates a binary mask for text using PIL"""
        W, H = size
        img = Image.new('L', size, 0)
        draw = ImageDraw.Draw(img)
        
        # Word wrap if needed
        if max_width > 0:
            text = textwrap.fill(text, width=max_width)
        
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Calculate position
        if align == "left":
            x = margin
        elif align == "right":
            x = W - text_w - margin
        else:  # center
            x = (W - text_w) // 2
            
        if position == "top":
            y = margin
        elif position == "bottom":
            y = H - text_h - margin
        else:  # middle
            y = (H - text_h) // 2
            
        # Draw text
        draw.text((x, y), text, font=font, fill=255)
        
        return np.array(img) / 255.0

    def _apply_animation_and_colors(self, output, mask, start_frame, end_frame,
                                  animation_type, duration, text_color, bg_color, opacity):
        """Applies animation and colors to the text mask"""
        
        # Convert colors to RGB tensors
        text_rgb = torch.tensor(self._hex_to_rgb(text_color), device=output.device) / 255.0
        bg_rgb = torch.tensor(self._hex_to_rgb(bg_color), device=output.device) / 255.0
        
        for frame in range(start_frame, end_frame + 1):
            if frame >= len(output):
                break
                
            # Calculate animation factor
            if animation_type == "none":
                factor = 1.0
            elif animation_type == "fade":
                factor = self._get_fade_factor(frame, start_frame, end_frame, duration)
            elif animation_type == "pop":
                factor = self._get_pop_factor(frame, start_frame, end_frame, duration)
            else:  # slide
                factor = self._get_slide_factor(frame, start_frame, end_frame, duration)
            
            # Apply animation and colors
            alpha = mask * factor * opacity
            output[frame, ..., :3] = (text_rgb * alpha[..., None] + 
                                    bg_rgb * (1 - alpha[..., None]))
            output[frame, ..., 3] = alpha

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Converts hex color to RGB values"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def _get_fade_factor(frame, start, end, duration):
        """Calculate fade animation factor"""
        if frame < start + duration:
            return min(1.0, (frame - start) / duration)
        elif frame > end - duration:
            return max(0.0, (end - frame) / duration)
        return 1.0

    @staticmethod
    def _get_pop_factor(frame, start, end, duration):
        """Calculate pop animation factor"""
        if frame < start + duration:
            t = (frame - start) / duration
            return min(1.0, 1.2 * (1 - (1 - t) ** 2))  # Overshoot and settle
        elif frame > end - duration:
            t = (end - frame) / duration
            return max(0.0, t)
        return 1.0

    @staticmethod
    def _get_slide_factor(frame, start, end, duration):
        """Calculate slide animation factor"""
        if frame < start + duration:
            return min(1.0, (frame - start) / duration)
        elif frame > end - duration:
            return max(0.0, (end - frame) / duration)
        return 1.0

    def _get_font_path(self, font_name):
        """Get built-in font path"""
        current_dir = Path(__file__).parent
        fonts_dir = current_dir / "fonts"
        return fonts_dir / self.BUILTIN_FONTS[font_name]




