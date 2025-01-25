from .features import WhisperFeature, BaseFeature
from .feature_extractors import FeatureExtractorBase
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ...tooltips import apply_tooltips

#NOTE: below is an example of the data we are expecting. 
# This is built to work with ComfyUI-Whisper, but only the json structure below is required.
#   # text:
    # We're telling a story about water and then about fire and then we're going to do ice and then psychedelic awesomeness. Okay.

    # #segement_alignment
    # [{"value": "We're telling a story about water and then about fire and then we're going to do ice and then", "start": 0.0, "end": 6.26}, {"value": "psychedelic awesomeness. Okay.", "start": 7.9799999999999995, "end": 9.38}]

    # #words_alignment
    # [{"value": "We're", "start": 0.0, "end": 0.26}, {"value": "telling", "start": 0.26, "end": 0.46}, {"value": "a", "start": 0.46, "end": 0.6}, {"value": "story", "start": 0.6, "end": 0.88}, {"value": "about", "start": 0.88, "end": 1.14}, {"value": "water", "start": 1.14, "end": 1.58}, {"value": "and", "start": 1.58, "end": 2.24}, {"value": "then", "start": 2.24, "end": 2.44}, {"value": "about", "start": 2.44, "end": 3.24}, {"value": "fire", "start": 3.24, "end": 3.66}, {"value": "and", "start": 3.66, "end": 4.1}, {"value": "then", "start": 4.1, "end": 4.3}, {"value": "we're", "start": 4.3, "end": 4.7}, {"value": "going", "start": 4.7, "end": 4.88}, {"value": "to", "start": 4.88, "end": 5.02}, {"value": "do", "start": 5.02, "end": 5.18}, {"value": "ice", "start": 5.18, "end": 5.44}, {"value": "and", "start": 5.44, "end": 5.98}, {"value": "then", "start": 5.98, "end": 6.26}, {"value": "psychedelic", "start": 7.9799999999999995, "end": 8.52}, {"value": "awesomeness.", "start": 8.52, "end": 9.06}, {"value": "Okay.", "start": 9.2, "end": 9.38}]

_category = f"{FeatureExtractorBase.CATEGORY}/Whisper"

class TriggerSet:
    def __init__(self):
        self.triggers = []
        self._expected_image_shape = None  # Store the expected shape for validation
    
    def add_trigger(self, pattern: str, values: tuple, mode: str, fade: str, 
                   duration: int, blend_mode: str, fill_behavior: str, image=None):
        # Validate image shape if provided
        if image is not None:
            current_shape = tuple(image.shape[1:])  # HWC shape (ignore batch dimension)
            
            if self._expected_image_shape is None:
                # First image sets the expected shape
                self._expected_image_shape = current_shape
            elif current_shape != self._expected_image_shape:
                raise ValueError(
                    f"Image shape mismatch! Expected shape {self._expected_image_shape}, "
                    f"but got {current_shape}. All trigger images must have the same dimensions."
                )
        
        trigger = {
            "pattern": pattern,
            "values": values,
            "mode": mode,
            "fade": fade,
            "duration": duration,
            "blend_mode": blend_mode,
            "fill_behavior": fill_behavior,
            "image": image,
            "image_batch_size": image.shape[0] if image is not None else 1
        }
        self.triggers.append(trigger)
        return self

    def extend(self, other_trigger_set):
        if other_trigger_set:
            # Validate image shapes from other trigger set
            for trigger in other_trigger_set.triggers:
                if trigger.get("image") is not None:
                    current_shape = tuple(trigger["image"].shape[1:])
                    if self._expected_image_shape is None:
                        self._expected_image_shape = current_shape
                    elif current_shape != self._expected_image_shape:
                        raise ValueError(
                            f"Image shape mismatch in trigger set combination! "
                            f"Expected shape {self._expected_image_shape}, but got {current_shape}. "
                            f"All trigger images must have the same dimensions."
                        )
            
            self.triggers.extend(other_trigger_set.triggers)
        return self

@apply_tooltips
class WhisperFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return WhisperFeature

    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (WhisperFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "alignment_data": ("whisper_alignment",),
            },
            "optional": {
                "trigger_set": ("TRIGGER_SET",),
                "context_size": ("INT", {"default": +3, "min": 0, "max": 10}),
                "overlap_mode": (["blend", "replace", "add"], {"default": "blend"}),
            }
        }

    RETURN_TYPES = ("FEATURE", "IMAGE")
    FUNCTION = "create_feature"
    CATEGORY = _category

    def create_feature(self, frame_rate, frame_count, width, height, extraction_method, 
                      alignment_data, trigger_set=None, context_size=3, overlap_mode="blend"):
        
        # Parse alignment data
        try:
            data = alignment_data if not isinstance(alignment_data, str) else json.loads(alignment_data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid alignment data format: {e}")
        
        # Parse trigger set if provided
        triggers = None
        if trigger_set:
            try:
                triggers = json.loads(trigger_set) if isinstance(trigger_set, str) else trigger_set
            except json.JSONDecodeError:
                print("Warning: Could not parse trigger_set JSON")
        
        # Create the WhisperFeature with the requested extraction method
        whisper_feature = WhisperFeature(
            name="whisper_feature",
            frame_rate=frame_rate,
            frame_count=frame_count,
            alignment_data=alignment_data,
            trigger_pairs=trigger_set,
            feature_name=extraction_method,
            width=width,
            height=height
        )
        
        whisper_feature.extract()
        
        # Process images if triggers contain them
        output_images = None
        if trigger_set and any(t.get("image") is not None for t in trigger_set.triggers):
            try:
                print(f"Processing {len(trigger_set.triggers)} triggers with images")
                image_sequence = []
                current_frame = 0
                
                sorted_triggers = whisper_feature.sort_triggers_by_occurrence(trigger_set.triggers)
                print(f"Sorted triggers: {len(sorted_triggers)}")
                
                # Instead of processing trigger by trigger, let's collect all occurrences first
                all_occurrences = []
                for trigger in sorted_triggers:
                    if trigger.get("image") is not None:
                        frame_ranges = whisper_feature.find_all_trigger_frames(trigger["pattern"])
                        for start_frame, end_frame in frame_ranges:
                            all_occurrences.append({
                                "trigger": trigger,
                                "start_frame": start_frame,
                                "end_frame": end_frame
                            })
                
                # Sort all occurrences by start frame to maintain chronological order
                all_occurrences.sort(key=lambda x: x["start_frame"])
                print(f"Processing {len(all_occurrences)} total trigger occurrences in chronological order")
                
                # Process occurrences in order
                for occurrence in all_occurrences:
                    trigger = occurrence["trigger"]
                    start_frame = occurrence["start_frame"]
                    end_frame = occurrence["end_frame"]
                    
                    # Convert back to tensor
                    images = torch.tensor(trigger["image"]) if isinstance(trigger["image"], list) else trigger["image"]
                    batch_size = trigger["image_batch_size"]
                    frame_count = end_frame - start_frame
                    
                    # Handle batch images based on fill behavior
                    if trigger["fill_behavior"] == "none":
                        frame_count = 1
                        images = images[:1]
                    else:
                        # Handle fill behaviors
                        if batch_size > 1:
                            if trigger["fill_behavior"] == "loop":
                                repeats = (frame_count + batch_size - 1) // batch_size
                                images = images.repeat((repeats, 1, 1, 1))[:frame_count]
                            elif trigger["fill_behavior"] == "hold":
                                if frame_count > batch_size:
                                    last_frame = images[-1:]
                                    extra_frames = frame_count - batch_size
                                    images = torch.cat([images, last_frame.repeat((extra_frames, 1, 1, 1))])
                        else:
                            if frame_count > 1:
                                images = images.repeat((frame_count, 1, 1, 1))
                    
                    if trigger["fill_behavior"] != "none":
                        # Only fill gaps if not "none"
                        if current_frame < start_frame:
                            zero_frames = start_frame - current_frame
                            zero_image = torch.zeros_like(images[0])
                            image_sequence.extend([zero_image] * zero_frames)
                    
                    image_sequence.extend(images.unbind(0))
                    current_frame = end_frame if trigger["fill_behavior"] != "none" else start_frame + 1
                
                if image_sequence:
                    print(f"Final sequence length: {len(image_sequence)}")
                    output_images = torch.stack(image_sequence)
                else:
                    print("No images in sequence")
                    
            except Exception as e:
                print(f"Error processing trigger images: {e}")
                import traceback
                traceback.print_exc()
                output_images = None
        
        # Set feature back to requested extraction method
        whisper_feature.set_active_feature(extraction_method)
        
        return (whisper_feature, output_images) if output_images is not None else (whisper_feature, None)

@apply_tooltips
class TriggerBuilder:
    """Creates triggers that respond to specific words or phrases in the Whisper transcription.
    Can be chained together to create complex trigger combinations.
    
    Example Usage:
        1. Single trigger for "amazing":
           - pattern: "amazing"
           - start_value: 0.0
           - end_value: 1.0
           
        2. Chain multiple triggers:
           TriggerBuilder("happy") -> TriggerBuilder("sad")
           Each trigger specifies how it blends with the accumulated result
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern": ("STRING", {"multiline": False, "default": "hello"}),
                "start_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "match_mode": (["exact", "contains", "regex", "phonetic"], {"default": "contains"}),
                "fade_type": (["none", "linear", "smooth"], {"default": "linear"}),
                "duration_frames": ("INT", {"default": 0, "min": 0}),
                "blend_mode": (["blend", "add", "multiply", "max"], {"default": "blend"}),
                "fill_behavior": (["none", "hold", "loop"], {"default": "none"}),
            },
            "optional": {
                "previous_triggers": ("TRIGGER_SET",),
                "trigger_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("TRIGGER_SET",)
    FUNCTION = "build"
    CATEGORY = _category

    def build(self, pattern: str, start_value: float, end_value: float, 
             match_mode: str, fade_type: str, duration_frames: int,
             blend_mode: str, fill_behavior: str, previous_triggers=None, 
             trigger_image=None) -> tuple:
        
        trigger_set = TriggerSet()
        
        # Add new trigger
        trigger_set.add_trigger(
            pattern=pattern,
            values=(start_value, end_value),
            mode=match_mode,
            fade=fade_type,
            duration=duration_frames,
            blend_mode=blend_mode,
            fill_behavior=fill_behavior,
            image=trigger_image
        )
        
        # Combine with previous triggers if any
        trigger_set.extend(previous_triggers)
        
        return (trigger_set,)

@apply_tooltips
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
    CATEGORY = _category

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
@apply_tooltips
class WhisperToPromptTravel:
    """Converts Whisper alignment data to prompt travel format"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alignment_data": ("whisper_alignment",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 120.0}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = _category

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

@apply_tooltips
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
    CATEGORY = _category

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

@apply_tooltips
class ManualWhisperAlignmentData:
    """Creates alignment data from manually entered text and timings.
    
    You can enter either word alignments or segment alignments.
    Format should match ComfyUI-Whisper output.
    
    Note: For proper whisper alignment data, you should use ComfyUI-Whisper:
    https://github.com/yuvraj108c/ComfyUI-Whisper
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "alignment_text": ("STRING", {
                    "multiline": True, 
                    "default": """
[{"value": "This is a manual alignment node. You should use ComfyUI-Whisper instead", "start": 0.0, "end": 6.26}, {"value": "for proper speech-to-text with timing.   https://github.com/yuvraj108c/ComfyUI-Whisper", "start": 7.98, "end": 9.38}]
"""
                }),
            }
        }
    
    RETURN_TYPES = ("whisper_alignment",)
    RETURN_NAMES = ("alignment_data",)
    FUNCTION = "parse"
    CATEGORY = _category

    def parse(self, alignment_text: str) -> tuple:        
        return (alignment_text,)

@apply_tooltips
class WhisperTimeAdjuster:
    """Manually adjust timing in Whisper alignment data"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alignment_data": ("whisper_alignment",),
                "time_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -1000.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "description": "Seconds to shift all timestamps (positive = delay, negative = earlier)"
                }),
            }
        }
    
    RETURN_TYPES = ("whisper_alignment",)
    FUNCTION = "adjust"
    CATEGORY = _category

    def adjust(self, alignment_data, time_offset):
        try:
            data = alignment_data if not isinstance(alignment_data, str) else json.loads(alignment_data)
            
            # Adjust all timestamps by offset
            adjusted_data = []
            for segment in data:
                adjusted_segment = segment.copy()
                adjusted_segment["start"] = segment["start"] + time_offset
                adjusted_segment["end"] = segment["end"] + time_offset
                adjusted_data.append(adjusted_segment)
            
            return (adjusted_data,)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid alignment data format: {e}")

@apply_tooltips
class WhisperAutoAdjust:
    """Automatically adjusts Whisper timing by detecting meaningful audio start"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alignment_data": ("whisper_alignment",),
                "audio": ("AUDIO",),  # Video Helper Suite audio format
                "detection_window": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 8192,
                    "step": 256,
                    "description": "Window size for energy detection"
                }),
                "energy_threshold": ("FLOAT", {
                    "default": 0.0001,
                    "min": 0.00001,
                    "max": 0.001,
                    "step": 0.00001,
                    "description": "Energy threshold (lower = more sensitive)"
                })
            }
        }
    
    RETURN_TYPES = ("whisper_alignment",)
    FUNCTION = "adjust"
    CATEGORY = _category

    def detect_audio_start(self, audio_dict, window_size, threshold):
        # Extract waveform and convert to mono numpy array, like BaseAudioProcessor
        audio = audio_dict['waveform'].squeeze(0).mean(axis=0).cpu().numpy()
        sample_rate = audio_dict['sample_rate']
        
        print(f"Audio shape before processing: {audio_dict['waveform'].shape}")
        print(f"Audio shape after mono conversion: {audio.shape}")
        print(f"Sample rate: {sample_rate}")
        
        # Calculate energy in windows using numpy for efficiency
        num_windows = (len(audio) - window_size) // (window_size // 2)
        energies = np.zeros(num_windows)
        
        for i in range(num_windows):
            start = i * (window_size // 2)
            end = start + window_size
            window = audio[start:end]
            energies[i] = np.mean(window ** 2)
        
        print(f"Energy shape: {energies.shape}")
        print(f"Max energy: {energies.max():.6f}")
        print(f"Min energy: {energies.min():.6f}")
        
        # Find first window above threshold
        start_window = np.where(energies > threshold)[0]
        if len(start_window) == 0:
            print(f"No windows above threshold {threshold}")
            return 0
        
        # Convert window index to sample
        start_sample = start_window[0] * (window_size // 2)
        start_time = start_sample / sample_rate
        
        print(f"Found start time: {start_time:.3f}s")
        return start_time

    def adjust(self, alignment_data, audio, detection_window=1024, energy_threshold=0.0001):
        try:
            data = alignment_data if not isinstance(alignment_data, str) else json.loads(alignment_data)
            
            print("Original whisper timings:")
            for segment in data[:3]:  # Print first 3 segments
                print(f"  {segment['start']:.3f} -> {segment['end']:.3f}: {segment['value']}")
            
            # Detect meaningful audio start
            audio_start = self.detect_audio_start(audio, detection_window, energy_threshold)
            print(f"\nDetected audio start: {audio_start}")
            
            # Get first whisper timestamp
            whisper_start = min(segment["start"] for segment in data)
            print(f"First whisper timestamp: {whisper_start}")
            
            # Calculate needed offset
            time_offset = audio_start - whisper_start
            print(f"Calculated offset: {time_offset}")
            
            # Adjust all timestamps
            adjusted_data = []
            for segment in data:
                adjusted_segment = segment.copy()
                adjusted_segment["start"] = segment["start"] + time_offset
                adjusted_segment["end"] = segment["end"] + time_offset
                adjusted_data.append(adjusted_segment)
            
            print("\nAdjusted whisper timings:")
            for segment in adjusted_data[:3]:  # Print first 3 segments
                print(f"  {segment['start']:.3f} -> {segment['end']:.3f}: {segment['value']}")
            
            return (adjusted_data,)
            
        except Exception as e:
            print(f"Error during adjustment: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error during alignment: {str(e)}")




