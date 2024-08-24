from .node_configs import CombinedMeta
from collections import OrderedDict
class RyanOnTheInside(metaclass=CombinedMeta):
    @classmethod
    def get_description(cls):
        
        # Get the display name from NODE_DISPLAY_NAME_MAPPINGS, fallback to cls.__name__ if not found
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(cls.__name__, cls.__name__)
        footer = "For more information, visit [RyanOnTheInside GitHub](https://github.com/ryanontheinside)."
        # Remove the " | RyanOnTheInside" suffix if present
        display_name = display_name.replace(" | RyanOnTheInside", "")
        
        desc = f"# {display_name}\n\n"
        
        if hasattr(cls, 'DESCRIPTION'):
            desc += f"{cls.DESCRIPTION}\n\n{footer}"
            return desc

        if hasattr(cls, 'TOP_DESCRIPTION'):
            desc += f"{cls.TOP_DESCRIPTION}\n\n"
        
        if hasattr(cls, "BASE_DESCRIPTION"):
            desc += cls.BASE_DESCRIPTION + "\n\n"
        
        # Use OrderedDict to maintain order and avoid duplication
        additional_info = OrderedDict()
        for c in cls.mro()[::-1]:  # Reverse method resolution order
            if hasattr(c, 'ADDITIONAL_INFO'):
                info = c.ADDITIONAL_INFO.strip()
                # Use the class name as a key to avoid duplication
                additional_info[c.__name__] = info
        
        # Join all unique ADDITIONAL_INFO entries
        if additional_info:
            desc += "\n\n".join(additional_info.values()) + "\n\n"
        
        if hasattr(cls, 'BOTTOM_DESCRIPTION'):
            desc += f"{cls.BOTTOM_DESCRIPTION}\n\n"

        desc += footer
        return desc
    
from .nodes.masks.temporal_masks import (
    MaskMorph,
    MaskTransform,
    MaskMath,
    MaskRings,
    MaskWarp,
    ) 

from .nodes.audio.audio_nodes import (
    AudioSeparator, 
    AudioFeatureVisualizer,
    FrequencyFilterCustom,
    FrequencyFilterPreset,
    AudioFilter,
    
    
)

from .nodes.flex.feature_extractors import(
    MIDILoadAndExtract,
    AudioFeatureExtractor,
    TimeFeatureNode,
    DepthFeatureNode,
)

from .nodes.masks.flex_masks import (
    FlexMaskMorph,
    FlexMaskWarp,
    FlexMaskTransform,
    FlexMaskMath
)
from .nodes.masks.optical_flow_masks import (
    OpticalFlowMaskModulation,
    OpticalFlowDirectionMask,
    OpticalFlowParticleSystem,
    )

from .nodes.masks.particle_system_masks import (
    ParticleEmissionMask,
    Vortex,
    GravityWell,
    ParticleEmitter,
    EmitterMovement,
    SpringJointSetting,
    StaticBody,
    ParticleColorModulation,
    ParticleSizeModulation,
    ParticleSpeedModulation,
    )

from .nodes.images.image_utility_nodes import DyeImage, ColorPicker

from .nodes.masks.utility_nodes import _mfc, TextMaskNode, MovingShape
import os
import folder_paths

# Get the directory of the current file
current_dir = os.path.dirname(os.path.realpath(__file__))

# Register the midi_files directory
midi_path = os.path.join(current_dir, "data/midi_files")
folder_paths.add_model_folder_path("midi_files", midi_path)

# Ensure the MIDI files directory exists
os.makedirs(midi_path, exist_ok=True)

NODE_CLASS_MAPPINGS = {
    "MaskMorph": MaskMorph,
    "MaskTransform":MaskTransform,
    "MaskMath":MaskMath,
    "MaskRings":MaskRings,
    "MaskWarp":MaskWarp,

    
    "OpticalFlowMaskModulation": OpticalFlowMaskModulation,
    "OpticalFlowParticleSystem":OpticalFlowParticleSystem,
    #"OpticalFlowDirectionMask":OpticalFlowDirectionMask,


    "ParticleEmissionMask":ParticleEmissionMask,
    "Vortex":Vortex,
    "GravityWell":GravityWell,
    "EmitterMovement":EmitterMovement,
    "ParticleEmitter":ParticleEmitter,
    "SpringJointSetting":SpringJointSetting,
    "StaticBody":StaticBody,
    "ParticleColorModulation":ParticleColorModulation,
    "ParticleSizeModulation":ParticleSizeModulation,
    "ParticleSpeedModulation":ParticleSpeedModulation,


    "FlexMaskMorph":       FlexMaskMorph,
    "FlexMaskWarp":        FlexMaskWarp,
    "FlexMaskTransform":   FlexMaskTransform,
    "FlexMaskMath":        FlexMaskMath,
    "AudioSeparator": AudioSeparator,
    "AudioFeatureExtractor":AudioFeatureExtractor,
    #"AudioFeatureVisualizer":AudioFeatureVisualizer,
    "FrequencyFilterCustom": FrequencyFilterCustom,
    "FrequencyFilterPreset": FrequencyFilterPreset,
    "AudioFilter":AudioFilter,


    "MIDILoadAndExtract":MIDILoadAndExtract,
    "TimeFeatureNode":TimeFeatureNode,
    "DepthFeatureNode": DepthFeatureNode    ,

    "MovingShape": MovingShape,
    "_mfc":_mfc,
    "TextMaskNode":TextMaskNode,
    "ColorPicker": ColorPicker,
    "DyeImage": DyeImage,
}

WEB_DIRECTORY = "./web/js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskMorph": "Temporal Mask Morph",
    "MaskTransform":"Temporal Mask Transform",
    "MaskMath":"Temporal Mask Math",
    "MaskRings":"Temporal Mask Rings",
    "MaskWarp":"Temporal Mask Warp",

    
    "OpticalFlowMaskModulation": "Optical Flow Mask Modulation",
    "OpticalFlowParticleSystem":"Optical Flow Particle System",
    #"OpticalFlowDirectionMask":"Optical Flow Direction Mask",
    
    "ParticleEmissionMask":"Particle Emission Mask",
    "Vortex": "Vortex",
    "GravityWell":"Gravity Well",
    "ParticleEmitter": "Particle Emitter",
    "EmitterMovement":"Emitter Movement",
    "SpringJointSetting":"Spring Joint Setting",
    "StaticBody":"Static Body",
    "ParticleColorModulation":"Particle Color Modulation",
    "ParticleSizeModulation": "Particle Size Modulation",
    "ParticleSpeedModulation":"Particle Speed Modulation",

    "AudioMaskMorph":"Audio Mask Morph",
    "AudioMaskWarp": "Audio Mask Warp",
    "AudioMaskTransform":"Audio Mask Transform",
    "AudioMaskMath": "Audio Mask Math",
    "AudioSeparator": "Audio Separator",

    #"AudioFeatureVisualizer": "Audio Feature Visualizer" ,
    "Frequency Filter Custom": "Frequency Filter Custom",
    "Frequency Filter Preset": "Frequency Filter Preset",
    "AudioFilter": "Audio Filter",
    
    "MIDILoadAndExtract":   "MIDI Load And Feature Extract",
    "AudioFeatureExtractor": "Audio Feature Extractor",
    "TimeFeatureNode":          "Time Feature Node ",
    "DepthFeatureNode":"Depth Feature Node",
    "MovingShape": "Moving Shape",
    "TextMaskNode":"Text Mask Node",
    "DyeImage" : "Dye Image"
}

suffix = " | RyanOnTheInside"
NODE_DISPLAY_NAME_MAPPINGS = {
    key: value if value.endswith(suffix) else value + suffix
    for key, value in NODE_DISPLAY_NAME_MAPPINGS.items()
}


from aiohttp import web
from server import PromptServer
from pathlib import Path

if hasattr(PromptServer, "instance"):

    # NOTE: we add an extra static path to avoid comfy mechanism
    # that loads every script in web. KJNodes
    PromptServer.instance.app.add_routes(
        [web.static("/ryanontheinside_web_async", (Path(__file__).parent.absolute() / "ryanontheinside_web_async").as_posix())]
    )


for node_name, node_class in NODE_CLASS_MAPPINGS.items():
    if hasattr(node_class, 'get_description'):
        desc = node_class.get_description()
        node_class.DESCRIPTION = desc
