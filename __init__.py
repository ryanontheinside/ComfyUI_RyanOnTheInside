from .node_configs import CombinedMeta
from collections import OrderedDict

#allows for central management and inheritance of class variables for help documentation
class RyanOnTheInside(metaclass=CombinedMeta):
    @classmethod
    def get_description(cls):
        
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(cls.__name__, cls.__name__)
        footer = "For more information, visit [RyanOnTheInside GitHub](https://github.com/ryanontheinside)."
        display_name = display_name.replace(" | RyanOnTheInside", "")
        
        desc = f"# {display_name}\n\n"
        
        if hasattr(cls, 'DESCRIPTION'):
            desc += f"{cls.DESCRIPTION}\n\n{footer}"
            return desc

        if hasattr(cls, 'TOP_DESCRIPTION'):
            desc += f"{cls.TOP_DESCRIPTION}\n\n"
        
        if hasattr(cls, "BASE_DESCRIPTION"):
            desc += cls.BASE_DESCRIPTION + "\n\n"
        
        additional_info = OrderedDict()
        for c in cls.mro()[::-1]:  
            if hasattr(c, 'ADDITIONAL_INFO'):
                info = c.ADDITIONAL_INFO.strip()
                
                additional_info[c.__name__] = info
        
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
    DownloadOpenUnmixModel,
    AudioFeatureVisualizer,
    FrequencyFilterCustom,
    FrequencyFilterPreset,
    AudioFilter,
    
)

from .nodes.flex.feature_extractors import(
    
    AudioFeatureExtractor,
    TimeFeatureNode,
    DepthFeatureNode,
    ColorFeatureNode,
    BrightnessFeatureNode,
    MotionFeatureNode,
)

from .nodes.flex.midi_feature_extractor import(
    MIDILoadAndExtract,
)

from .nodes.masks.flex_masks import (
    FlexMaskMorph,
    FlexMaskWarp,
    FlexMaskTransform,
    FlexMaskMath,
    FlexMaskOpacity,
    FlexMaskVoronoi,
    FlexMaskVoronoiScheduled,
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

from .nodes.masks.opacity_masks import(
    DepthBasedMaskOpacity,
    FlexDepthBasedMaskOpacity,
)

from .nodes.masks.utility_nodes import (
    _mfc, 
    TextMaskNode, 
    MovingShape,
)

###images

from .nodes.images.image_utility_nodes import (
    DyeImage,
)
from .nodes.images.flex_images import (
    FlexImageEdgeDetect,
    FlexImagePosterize,
    FlexImageKaleidoscope,
    FlexImageBloom,
    FlexImageChromaticAberration,
    FlexImageGlitch,
    FlexImagePixelate,
    FlexImageColorGrade,
    FlexImageTiltShift,
)

from .nodes.flex.feature_externals import (
    FeatureToWeightsStrategy,
)

from .nodes.flex.feature_modulation import (
    FeatureMixer
)

from .nodes.five.adaptive_lora import (
    LoraWeightStrategyLoader
)


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
    
    ###temporal
    "MaskMorph": MaskMorph,
    "MaskTransform":MaskTransform,
    "MaskMath":MaskMath,
    "MaskRings":MaskRings,
    "MaskWarp":MaskWarp,

    
    #optical flow
    "OpticalFlowMaskModulation": OpticalFlowMaskModulation,
    "OpticalFlowParticleSystem":OpticalFlowParticleSystem,
    "OpticalFlowDirectionMask":OpticalFlowDirectionMask,

    #particle simulation
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


    #flex masks
    "FlexMaskMorph":       FlexMaskMorph,
    "FlexMaskWarp":        FlexMaskWarp,
    "FlexMaskTransform":   FlexMaskTransform,
    "FlexMaskMath":        FlexMaskMath,

    #audio
    "AudioSeparator": AudioSeparator,
    "DownloadOpenUnmixModel":DownloadOpenUnmixModel,
    "AudioFeatureVisualizer":AudioFeatureVisualizer,
    "FrequencyFilterCustom": FrequencyFilterCustom,
    "FrequencyFilterPreset": FrequencyFilterPreset,
    "AudioFilter":AudioFilter,

    #features
    "AudioFeatureExtractor":AudioFeatureExtractor,
    "MIDILoadAndExtract":MIDILoadAndExtract,
    "TimeFeatureNode":TimeFeatureNode,
    "DepthFeatureNode": DepthFeatureNode,
    "ColorFeatureNode":ColorFeatureNode,
    "BrightnessFeatureNode":BrightnessFeatureNode,
    "MotionFeatureNode":MotionFeatureNode,
    "FeatureToWeightsStrategy": FeatureToWeightsStrategy,
    "FeatureMixer":FeatureMixer,
    
    'FlexImageEdgeDetect':FlexImageEdgeDetect,
    "FlexImagePosterize":FlexImagePosterize,
    "FlexImageKaleidoscope":FlexImageKaleidoscope,
    "FlexImageBloom":FlexImageBloom,
    "FlexImageChromaticAberration":FlexImageChromaticAberration,
    "FlexImageGlitch":FlexImageGlitch,
    "FlexImagePixelate":FlexImagePixelate,
    "FlexImageColorGrade":FlexImageColorGrade,
    "FlexImageTiltShift":FlexImageTiltShift,
    "FlexMaskOpacity":FlexMaskOpacity,
    "FlexMaskVoronoi":FlexMaskVoronoi,
    "FlexMaskVoronoiScheduled":FlexMaskVoronoiScheduled,
    #opacity
    "FlexDepthBasedMaskOpacity":FlexDepthBasedMaskOpacity,
    "DepthBasedMaskOpacity":DepthBasedMaskOpacity,


    #garb
    "DyeImage": DyeImage,
    "MovingShape": MovingShape,
    "_mfc":_mfc,
    "TextMaskNode":TextMaskNode,
    


    "LoraWeightStrategyLoader": LoraWeightStrategyLoader


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

    "AudioFeatureVisualizer": "Audio Feature Visualizer" ,
    "Frequency Filter Custom": "Frequency Filter Custom",
    "Frequency Filter Preset": "Frequency Filter Preset",
    "AudioFilter": "Audio Filter",
    
    "MIDILoadAndExtract":   "MIDI Load & Feature Extract",
    "AudioFeatureExtractor": "Audio Feature & Extractor",
    "TimeFeatureNode":          "Time Feature",
    "DepthFeatureNode":"Depth Feature",
    "BrightnessFeatureNode":"Brightness Feature",
    "MotionFeatureNode":"Motion Feature",

    "MovingShape": "Moving Shape",
    "TextMaskNode":"Text Mask Node",


    "DyeImage" : "Dye Image",
    # "FlexImageAdjustment":"Flex Image Adjustment",
    # "FlexImageFilter":"Flex Image Filter",
    # "FlexImageBlend":"Flex Image Blend",
}




import re

suffix = " | RyanOnTheInside"

for node_name in NODE_CLASS_MAPPINGS.keys():
    if node_name not in NODE_DISPLAY_NAME_MAPPINGS:
        # Convert camelCase or snake_case to Title Case
        display_name = ' '.join(word.capitalize() for word in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', node_name))
    else:
        display_name = NODE_DISPLAY_NAME_MAPPINGS[node_name]
    
    # Add the suffix if it's not already present
    if not display_name.endswith(suffix):
        display_name += suffix
    
    NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name


from aiohttp import web
from server import PromptServer
from pathlib import Path

if hasattr(PromptServer, "instance"):

    # NOTE: we add an extra static path to avoid comfy mechanism
    # that loads every script in web. 
    # 
    # Again credit to KJNodes and MTB nodes

    PromptServer.instance.app.add_routes(
        [web.static("/ryanontheinside_web_async", (Path(__file__).parent.absolute() / "ryanontheinside_web_async").as_posix())]
    )



for node_name, node_class in NODE_CLASS_MAPPINGS.items():
    if hasattr(node_class, 'get_description'):
        desc = node_class.get_description()
        node_class.DESCRIPTION = desc
