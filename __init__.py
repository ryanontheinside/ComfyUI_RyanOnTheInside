from .node_configs.node_configs import CombinedMeta
from collections import OrderedDict

#allows for central management and inheritance of class variables for help documentation
class RyanOnTheInside(metaclass=CombinedMeta):
    @classmethod
    def get_description(cls):
        
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(cls.__name__, cls.__name__)
        footer = "For more information, visit [RyanOnTheInside GitHub](https://github.com/ryanontheinside).\n\n"
        footer += "For tutorials and example workflows visit [RyanOnTheInside Civitai](https://civitai.com/user/ryanontheinside).\n\n"
        footer += "For video tutorials and more visit [RyanOnTheInside YouTube](https://www.youtube.com/@ryanontheinside).\n\n"
        display_name = display_name.replace(" | RyanOnTheInside", "")
        
        desc = f"# {display_name}\n\n"
        
        if hasattr(cls, 'DESCRIPTION'):
            desc += f"{cls.DESCRIPTION}\n\n{footer}"
            return desc

        if hasattr(cls, 'TOP_DESCRIPTION'):
            desc += f"### {cls.TOP_DESCRIPTION}\n\n"
        
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
    # DownloadCREPEModel,
    AudioFeatureVisualizer,
    FrequencyFilterCustom,
    FrequencyFilterPreset,
    AudioFilter,
    EmptyMaskFromAudio,
    EmptyImageFromAudio,
    
)

from .nodes.flex.feature_extractors import(
    TimeFeatureNode,
    DepthFeatureNode,
    ColorFeatureNode,
    BrightnessFeatureNode,
    MotionFeatureNode,
    AreaFeatureNode,
)

from .nodes.flex.feature_extractors_audio import(
    AudioFeatureExtractor,
    PitchRangeNode,
    PitchRangePresetNode,
    PitchRangeByNoteNode,
    PitchFeatureExtractor,
)

from .nodes.flex.feature_extractors_midi import(
    MIDILoadAndExtract,
)

from .nodes.flex.feature_extractors_proximity import(
    LocationFromMask,
    ProximityFeatureNode,
    LocationFromPoint,
    LocationTransform,
)

from .nodes.flex.visualizers import(
    ProximityVisualizer,
    EffectVisualizer,
    PitchVisualizer,
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

from .nodes.masks.mask_utility_nodes import (
    _mfc, 
    TextMaskNode, 
    MovingShape,
)

from .nodes.utility_nodes import (
    ImageChunks, 
    ImageIntervalSelect,
    VideoChunks,
    ImageDifference,
    ImageShuffle,
    SwapDevice,
    ImageIntervalSelectPercentage,
)

###images

from .nodes.images.image_utility_nodes import (
    DyeImage,
)

from .nodes.masks.flex_masks import (
    FlexMaskMorph,
    FlexMaskWarp,
    FlexMaskTransform,
    FlexMaskMath,
    FlexMaskOpacity,
    FlexMaskVoronoiScheduled,
    FlexMaskBinary,
    FlexMaskWavePropagation,
    FlexMaskEmanatingRings,
    FlexMaskRandomShapes,
    FlexMaskDepthChamber,
   # FlexMaskDepthChamberRelative, #NOTE work in progress

)

from .nodes.masks.flex_masks_normal import (
    FlexMaskNormalLighting,

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
    FlexImageParallax,
    FlexImageContrast,
    
)

from .nodes.depth.depth_base import(
    DepthInjection,
    DepthBlender,
    DepthRippleEffect,
)

from .nodes.flex.feature_externals import (
    FeatureToWeightsStrategy,
    DepthShapeModifier,
)

from .nodes.flex.feature_modulation import (
    FeatureMixer,
    FeatureCombine,
    FeatureOscillator,
    FeatureScaler,
    FeatureSmoothing,
    FeatureFade,
    PreviewFeature,
    FeatureMath,
    FeatureRebase,
    FeatureTruncateOrExtend,
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
    "MaskMorph":                    MaskMorph,
    "MaskTransform":                MaskTransform,
    "MaskMath":                     MaskMath,
    "MaskRings":                    MaskRings,
    "MaskWarp":                     MaskWarp,


    #optical flow   
    "OpticalFlowMaskModulation":    OpticalFlowMaskModulation,
    "OpticalFlowParticleSystem":    OpticalFlowParticleSystem,
    "OpticalFlowDirectionMask":     OpticalFlowDirectionMask,

    #particle simulation    
    "ParticleEmissionMask":         ParticleEmissionMask,
    "Vortex":                       Vortex,
    "GravityWell":                  GravityWell,
    "EmitterMovement":              EmitterMovement,
    "ParticleEmitter":              ParticleEmitter,
    "SpringJointSetting":           SpringJointSetting,
    "StaticBody":                   StaticBody,
    "ParticleColorModulation":      ParticleColorModulation,
    "ParticleSizeModulation":       ParticleSizeModulation,
    "ParticleSpeedModulation":      ParticleSpeedModulation,


    #flex masks 
    "FlexMaskMorph":                FlexMaskMorph,
    "FlexMaskWarp":                 FlexMaskWarp,
    "FlexMaskTransform":            FlexMaskTransform,
    "FlexMaskMath":                 FlexMaskMath,
    "FlexMaskBinary":               FlexMaskBinary,
    "FlexMaskOpacity":              FlexMaskOpacity,
    "FlexMaskVoronoiScheduled":     FlexMaskVoronoiScheduled,
    "FlexMaskWavePropagation":      FlexMaskWavePropagation,
    "FlexMaskEmanatingRings":       FlexMaskEmanatingRings,
    "FlexMaskRandomShapes":         FlexMaskRandomShapes,
    "FlexMaskDepthChamber":         FlexMaskDepthChamber,
    #"FlexMaskNormalLighting":       FlexMaskNormalLighting,
    # "FlexMaskDepthChamberRelative": FlexMaskDepthChamberRelative,

    #audio  
    "AudioSeparator":               AudioSeparator,
    "DownloadOpenUnmixModel":       DownloadOpenUnmixModel,
    # "DownloadCREPEModel":           DownloadCREPEModel,
    "AudioFeatureVisualizer":       AudioFeatureVisualizer,
    "FrequencyFilterCustom":        FrequencyFilterCustom,
    "FrequencyFilterPreset":        FrequencyFilterPreset,
    "AudioFilter":                  AudioFilter,
    "EmptyMaskFromAudio":           EmptyMaskFromAudio,
    "EmptyImageFromAudio":          EmptyImageFromAudio,

    #features   
    "AudioFeatureExtractor":        AudioFeatureExtractor,
    "PitchFeatureExtractor":        PitchFeatureExtractor,
    "PitchRange":                   PitchRangeNode,
    "PitchRangePreset":             PitchRangePresetNode,
    "PitchRangeByNoteNode":         PitchRangeByNoteNode,
    "MIDILoadAndExtract":           MIDILoadAndExtract,
    "TimeFeatureNode":              TimeFeatureNode,
    "DepthFeatureNode":             DepthFeatureNode,
    "ColorFeatureNode":             ColorFeatureNode,
    "BrightnessFeatureNode":        BrightnessFeatureNode,
    "MotionFeatureNode":            MotionFeatureNode,
    "LocationFromMask":             LocationFromMask,
    "ProximityFeatureNode":         ProximityFeatureNode,
    "LocationFromPoint":            LocationFromPoint,
    "LocationTransform":            LocationTransform,
    "AreaFeatureNode":              AreaFeatureNode,

    "FeatureToWeightsStrategy":     FeatureToWeightsStrategy,
    "DepthInjection":               DepthInjection,
    "DepthRippleEffect":            DepthRippleEffect,
    "DepthBlender":                 DepthBlender,
    "DepthShapeModifier":           DepthShapeModifier,

    #feature modulation
    "FeatureMixer":                 FeatureMixer,
    "FeatureCombine":               FeatureCombine,
    "FeatureOscillator":            FeatureOscillator,
    "FeatureScaler":                FeatureScaler,
    "FeatureSmoothing":             FeatureSmoothing,
    "FeatureFade":                  FeatureFade,
    "FeatureMath":                  FeatureMath,
    "PreviewFeature":               PreviewFeature,
    "FeatureRebase":                FeatureRebase,
    "FeatureTruncateOrExtend":      FeatureTruncateOrExtend,

    #images
    'FlexImageEdgeDetect':          FlexImageEdgeDetect,
    "FlexImagePosterize":           FlexImagePosterize,
    "FlexImageKaleidoscope":        FlexImageKaleidoscope,
    "FlexImageBloom":               FlexImageBloom,
    "FlexImageChromaticAberration": FlexImageChromaticAberration,
    "FlexImageGlitch":              FlexImageGlitch,
    "FlexImagePixelate":            FlexImagePixelate,
    "FlexImageColorGrade":          FlexImageColorGrade,
    "FlexImageTiltShift":           FlexImageTiltShift,
    "FlexImageParallax":            FlexImageParallax,
    "FlexImageContrast":            FlexImageContrast,
    #opacity xp 
    # "FlexDepthBasedMaskOpacity":  FlexDepthBasedMaskOpacity,
    # "DepthBasedMaskOpacity":      DepthBasedMaskOpacity,

    #visulizers
    "ProximityVisualizer":          ProximityVisualizer,
    "EffectVisualizer":             EffectVisualizer,
    "PitchVisualizer":              PitchVisualizer,

    #garb   
    "DyeImage":                     DyeImage,
    "MovingShape":                  MovingShape,
    "_mfc":                         _mfc,
    "TextMaskNode":                 TextMaskNode,

    #utility nodes
    "ImageChunk":                   ImageChunks, 
    "ImageInterval":                ImageIntervalSelect,
    "VideoChunk":                   VideoChunks,
    "ImageDifference":              ImageDifference,
    "ImageShuffle":                 ImageShuffle,
    "SwapDevice":                   SwapDevice,
    "ImageIntervalSelectPercentage":ImageIntervalSelectPercentage,
    
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

  
    "AudioSeparator": "Audio Separator",

    "AudioFeatureVisualizer": "Audio Feature Visualizer" ,
    "Frequency Filter Custom": "Frequency Filter Custom",
    "Frequency Filter Preset": "Frequency Filter Preset",
    "AudioFilter": "Audio Filter",
  

    "MIDILoadAndExtract":   "MIDI Load & Feature Extract",
    "PitchRangeByNoteNode": "Pitch Range By Note",
    "AudioFeatureExtractor": "Audio Feature & Extractor",
    "TimeFeatureNode":          "Time Feature",
    "DepthFeatureNode":"Depth Feature",
    "BrightnessFeatureNode":"Brightness Feature",
    "MotionFeatureNode":"Motion Feature",

    "FeatureMixer":                 "FeatureMod Mixer",
    "FeatureCombine":               "FeatureMod Combine",
    "FeatureOscillator":            "FeatureMod Oscillator",
    "FeatureScaler":                "FeatureMod Scaler",
    "FeatureSmoothing":             "FeatureMod Smoothing",
    "FeatureMath":                  "FeatureMod Math",
    "MovingShape": "Moving Shape",
    "TextMaskNode":"Text Mask Node",


    "DyeImage" : "Dye Image",
    "ImageIntervalSelectPercentage":  "Image Interval Select %"
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
