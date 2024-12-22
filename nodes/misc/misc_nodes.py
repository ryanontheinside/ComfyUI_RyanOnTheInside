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
