# Set this to True to disable all tooltips
DISABLE_TOOLTIPS = False

if not DISABLE_TOOLTIPS:
    from .tooltips import TooltipManager, apply_tooltips
    from .tooltips.categories import register_all_tooltips

from comfy.utils import ProgressBar
from tqdm import tqdm
from .node_configs.node_configs import CombinedMeta
from collections import OrderedDict
import os
import folder_paths
import shutil

#NOTE: THIS IS LEGACY FOR BACKWARD COMPATIBILITY. FUNCTIONALLY REPLACED BY TOOLTIPS.
#NOTE: allows for central management and inheritance of class variables for help documentation
class RyanOnTheInside(metaclass=CombinedMeta):
    @classmethod
    def get_description(cls):
        return ""

class ProgressMixin:
    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)
        self.tqdm_bar = tqdm(total=total_steps, desc=desc, leave=False)
        self.current_progress = 0
        self.total_steps = total_steps

    def update_progress(self, step=1):
        self.current_progress += step
        if self.progress_bar:
            self.progress_bar.update(step)
        if self.tqdm_bar:
            self.tqdm_bar.update(step)

    def end_progress(self):
        if self.tqdm_bar:
            self.tqdm_bar.close()
        self.progress_bar = None
        self.tqdm_bar = None
        self.current_progress = 0
        self.total_steps = 0

print("""
     ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗    ██████╗ ███╗   ██╗
     ██╔══██╗╚██╗ ██╔╝██╔══██╗████╗  ██║   ██╔═══██╗████╗  ██║
     ██████╔╝ ╚████╔╝ ███████║██╔██╗ ██║   ██║   ██║██╔██╗ ██║
     ██╔══██╗  ╚██╔╝  ██╔══██║██║╚██╗██║   ██║   ██║██║╚██╗██║
     ██║  ██║   ██║   ██║  ██║██║ ╚████║   ╚██████╔╝██║ ╚████║
     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═════╝ ╚═╝  ╚═══╝
████████╗██╗  ██╗███████╗   ██╗███╗   ██╗███████╗██╗██████╗ ███████╗
╚══██╔══╝██║  ██║██╔════╝   ██║████╗  ██║██╔════╝██║██╔══██╗██╔════╝
   ██║   ███████║█████╗     ██║██╔██╗ ██║███████╗██║██║  ██║█████╗  
   ██║   ██╔══██║██╔══╝     ██║██║╚██╗██║╚════██║██║██║  ██║██╔══╝  
   ██║   ██║  ██║███████╗   ██║██║ ╚████║███████║██║██████╔╝███████╗
   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝╚═════╝ ╚══════╝

             ⚡ R Y A N   O N   T H E   I N S I D E ⚡

      """)


    
from .nodes.masks.temporal_masks import (
    MaskMorph,
    MaskTransform,
    MaskMath,
    MaskRings,
    MaskWarp,
    ) 

from .nodes.audio.audio_nodes import (
    AudioSeparatorSimple,
    DownloadOpenUnmixModel,
    # DownloadCREPEModel,
    AudioFeatureVisualizer,
    FrequencyFilterCustom,
    FrequencyFilterPreset,
    FrequencyRange,
    AudioFilter,
    EmptyMaskFromAudio,
    EmptyImageFromAudio,
    EmptyImageAndMaskFromAudio,

)

from .nodes.audio.flex_audio_visualizer import ( 
    FlexAudioVisualizerCircular,
    FlexAudioVisualizerLine,
    FlexAudioVisualizerContour,
)

from .nodes.audio.audio_nodes_effects import (
    AudioPitchShift,
    AudioTimeStretch,
    
    AudioGain,
    AudioFade,
)

from .nodes.audio.audio_nodes_utility import (
    AudioPad,
    AudioChannelMerge,
    AudioChannelSplit,
    AudioResample,
    AudioVolumeNormalization,
    Audio_Combine,
    AudioSubtract,
    Audio_Concatenate,
    AudioDither,
    AudioTrim,
    AudioInfo,
    Knob,
)

from .nodes.flex.feature_extractors import(
    TimeFeatureNode,
    DepthFeatureNode,
    ColorFeatureNode,
    BrightnessFeatureNode,
    MotionFeatureNode,
    AreaFeatureNode,
    ManualFeatureNode,
    ManualFeatureFromPipe,
    DrawableFeatureNode,
    FeatureInfoNode,
    FloatFeatureNode
)

from .nodes.flex.feature_extractors_whisper import( 
    WhisperFeatureNode,
    TriggerBuilder,   
    ContextModifier,
    WhisperToPromptTravel,
    WhisperTextRenderer,
    ManualWhisperAlignmentData,
    WhisperAutoAdjust,
    WhisperTimeAdjuster
)

from .nodes.flex.feature_extractors_audio import(
    AudioFeatureExtractor,
    PitchRangeNode,
    PitchRangePresetNode,
    PitchRangeByNoteNode,
    PitchFeatureExtractor,
    RhythmFeatureExtractor,
    
)

from .nodes.flex.feature_extractors_midi import(
    MIDIFeatureExtractor,
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
    PreviewFeature,
    AnimatedFeaturePreview,
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
    EmitterEmissionRateModulation,
    )

from .nodes.masks.mask_utility_nodes import (
    _mfc, 
    TextMaskNode, 
    MovingShape,
    MaskCompositePlus,
    AdvancedLuminanceMask,
    TranslucentComposite,
)

from .nodes.utility_nodes import (
    ImageChunks, 
    ImageIntervalSelect,
    VideoChunks,
    ImageDifference,
    Image_Shuffle,
    SwapDevice,
    ImageIntervalSelectPercentage,
    ImageIndexSelect,
)

###images

from .nodes.images.image_utility_nodes import (
    DyeImage,
    Color_Picker,
    ImageCASBatch,
    ImageScaleToTarget

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
    FlexMaskInterpolate,

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
    FlexImageWarp,
    FlexImageVignette,
    FlexImageTransform,
    FlexImageHueShift,
    FlexImageDepthWarp,
    FlexImageHorizontalToVertical,
)

from .nodes.video.flex_video import (
    FlexVideoSpeed,
    FlexVideoDirection,
    FlexVideoFrameBlend,
    FlexVideoSeek,
)

from .nodes.depth.depth_base import(
    DepthInjection,
    DepthBlender,
    DepthRippleEffect,
)

from .nodes.flex.flex_externals import (
    FeatureToWeightsStrategy,
    FeatureToSplineData,
    SplineFeatureModulator,
    SplineRhythmModulator,
    DepthShapeModifier,
    DepthShapeModifierPrecise,
    FeatureToFloat,
    FeatureToMask,
    FeatureToFilteredList,
)


from .nodes.flex.feature_modulation import (
    FeatureMixer,
    FeatureCombine,
    FeatureOscillator,
    FeatureScaler,
    FeatureSmoothing,
    FeatureFade,
    
    FeatureMath,
    FeatureRebase,
    FeatureTruncateOrExtend,
    FeatureAccumulate,
    FeatureContiguousInterpolate,
    FeatureRenormalize,
    FeatureInterpolator,
    FeaturePeakDetector,
    FeatureInterpolateMulti,
)

from .nodes.audio.flex_audio import (
    FlexAudioPitchShift,
    FlexAudioTimeStretch,
)

from .nodes.latents.flex_latents import (
    FlexLatentInterpolate,
    EmbeddingGuidedLatentInterpolate,
    FlexLatentBlend,
    FlexLatentNoise,
)
from .nodes.flex.parameter_scheduling import (
    FeatureToFlexIntParam,
    FeatureToFlexFloatParam,
)

from .nodes.latents.latent_frequency_blender import LatentFrequencyBlender

from .nodes.flex.feature_pipe import ManualFeaturePipe

from .nodes.preprocessors.pose import PoseInterpolator

from .nodes.doom.doom import Doom_

# from .nodes.models.flex_model_base import FlexFeatureAttentionControl

from .audio_latent_blend import AudioLatentBlend, FlexlatentAudioBlend


from .nodes.audio.midi_nodes import MIDIToAudio, MIDILoader

# Import external integrations
from .external_integration import (
    HAS_ADVANCED_LIVE_PORTRAIT,
    HAS_ADVANCED_CONTROLNET,
    HAS_ANIMATEDIFF,
    EXTERNAL_NODE_CLASS_MAPPINGS,
)


# Get the directory of the current file
current_dir = os.path.dirname(os.path.realpath(__file__))

# Register the midi_files directory
midi_path = os.path.join(current_dir, "data/midi_files")
folder_paths.add_model_folder_path("midi_files", midi_path)

# Ensure the MIDI files directory exists
os.makedirs(midi_path, exist_ok=True)

# Get the path to ComfyUI's web/extensions directory
extension_path = os.path.join(os.path.dirname(folder_paths.__file__), "web", "extensions")
my_extension_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web", "extensions")

# Create RyanOnTheInside subfolder in ComfyUI extensions
roti_extension_path = os.path.join(extension_path, "RyanOnTheInside")
os.makedirs(roti_extension_path, exist_ok=True)

# Clean up existing files in the RyanOnTheInside folder
for file in os.listdir(roti_extension_path):
    os.remove(os.path.join(roti_extension_path, file))

# Copy our extension files to ComfyUI's extensions/RyanOnTheInside directory
if os.path.exists(my_extension_path):
    for file in os.listdir(my_extension_path):
        if file.endswith('.js'):
            src = os.path.join(my_extension_path, file)
            dst = os.path.join(roti_extension_path, file)
            print(f"[RyanOnTheInside] Copying extension file: {file}")
            shutil.copy2(src, dst)
NODE_CLASS_MAPPINGS = {
    #NOTE: PoseInterpolator is not working yet
    #"PoseInterpolator": PoseInterpolator,
    # "FlexFeatureAttentionControl": FlexFeatureAttentionControl,
    "AudioLatentBlend":             AudioLatentBlend,
    "FlexlatentAudioBlend": FlexlatentAudioBlend,
    "FeatureInterpolator":          FeatureInterpolator,
    "FeaturePeakDetector":          FeaturePeakDetector,
    "FeatureInterpolateMulti":      FeatureInterpolateMulti,
    
    "Doom_": Doom_,
    "WhisperToPromptTravel":        WhisperToPromptTravel,
    "ManualFeaturePipe":            ManualFeaturePipe,
    "ManualFeatureFromPipe":        ManualFeatureFromPipe,
    #latents
    "FlexLatentInterpolate":        FlexLatentInterpolate,
    "EmbeddingGuidedLatentInterpolate": EmbeddingGuidedLatentInterpolate,
    "FlexLatentBlend":              FlexLatentBlend,
    "FlexLatentNoise":              FlexLatentNoise,
    "LatentFrequencyBlender":       LatentFrequencyBlender,
    #video
    "FlexVideoSpeed":               FlexVideoSpeed,
    "FlexVideoDirection":           FlexVideoDirection,
    "FlexVideoFrameBlend":          FlexVideoFrameBlend,
    "FlexVideoSeek":                FlexVideoSeek,
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
    "EmitterEmissionRateModulation":EmitterEmissionRateModulation,

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
    "FlexMaskInterpolate":          FlexMaskInterpolate,

    #flex audio
    "FlexAudioPitchShift":          FlexAudioPitchShift,
    "FlexAudioTimeStretch":         FlexAudioTimeStretch,

    #flex audio visualizers
    "FlexAudioVisualizerCircular":  FlexAudioVisualizerCircular,
    "FlexAudioVisualizerLine":      FlexAudioVisualizerLine,
    "FlexAudioVisualizerContour":   FlexAudioVisualizerContour,
    #audio  

    "AudioSeparatorSimple":         AudioSeparatorSimple,
    "DownloadOpenUnmixModel":       DownloadOpenUnmixModel,
    # "DownloadCREPEModel":           DownloadCREPEModel,
    "AudioFeatureVisualizer":       AudioFeatureVisualizer,
    "FrequencyFilterCustom":        FrequencyFilterCustom,
    "FrequencyFilterPreset":        FrequencyFilterPreset,
    "FrequencyRange":               FrequencyRange,
    "AudioFilter":                  AudioFilter,
    "EmptyMaskFromAudio":           EmptyMaskFromAudio,
    "EmptyImageFromAudio":          EmptyImageFromAudio,
    "EmptyImageAndMaskFromAudio":   EmptyImageAndMaskFromAudio,
    "Audio_Combine":                 Audio_Combine,
    "AudioSubtract":                AudioSubtract,
    "Audio_Concatenate":             Audio_Concatenate,
    "AudioPitchShift":              AudioPitchShift,
    "AudioTimeStretch":             AudioTimeStretch,
    "AudioDither":                  AudioDither,
    "AudioTrim":                    AudioTrim,
    "AudioInfo":                    AudioInfo,
    "AudioGain":                    AudioGain,
    "AudioFade":                    AudioFade,
    "AudioPad":                     AudioPad,
    "AudioChannelMerge":            AudioChannelMerge,
    "AudioChannelSplit":            AudioChannelSplit,
    "AudioResample":                AudioResample,
    "AudioVolumeNormalization":     AudioVolumeNormalization,
    "Knob":                Knob,

    #features   
    "AudioFeatureExtractor":        AudioFeatureExtractor,

#TODO make feature info JS display info
#TODO: support negative feature values for opposit direction......
    

    "PitchFeatureExtractor":        PitchFeatureExtractor,
    "RhythmFeatureExtractor":       RhythmFeatureExtractor,
    
    "PitchRange":                   PitchRangeNode,
    "PitchRangePreset":             PitchRangePresetNode,
    "PitchRangeByNoteNode":         PitchRangeByNoteNode,
    "MIDIFeatureExtractor":           MIDIFeatureExtractor,
    "TimeFeatureNode":              TimeFeatureNode,
    "FloatFeatureNode":             FloatFeatureNode,
    "ManualFeatureNode":            ManualFeatureNode,
    "ManualFeatureFromPipe":        ManualFeatureFromPipe,
    "DrawableFeatureNode":          DrawableFeatureNode,
    "DepthFeatureNode":             DepthFeatureNode,
    "ColorFeatureNode":             ColorFeatureNode,

    "BrightnessFeatureNode":        BrightnessFeatureNode,
    "MotionFeatureNode":            MotionFeatureNode,
    "LocationFromMask":             LocationFromMask,
    "ProximityFeatureNode":         ProximityFeatureNode,
    "LocationFromPoint":            LocationFromPoint,
    "LocationTransform":            LocationTransform,
    "AreaFeatureNode":              AreaFeatureNode,
    "FeatureInfoNode":              FeatureInfoNode,
    "WhisperFeature":               WhisperFeatureNode,
    "TriggerBuilder":               TriggerBuilder,
    "ContextModifier":              ContextModifier,
    "WhisperTextRenderer":          WhisperTextRenderer,
    "WhisperAutoAdjust":            WhisperAutoAdjust,
    "WhisperTimeAdjuster":          WhisperTimeAdjuster,
    "ManualWhisperAlignmentData":   ManualWhisperAlignmentData,

    "FeatureToWeightsStrategy":     FeatureToWeightsStrategy,
    "FeatureToSplineData":          FeatureToSplineData,
    "SplineFeatureModulator":       SplineFeatureModulator,
    "FeatureToFloat":               FeatureToFloat,
    "FeatureToMask":                 FeatureToMask,
    "FeatureToFilteredList":        FeatureToFilteredList,
    "SplineRhythmModulator":        SplineRhythmModulator,
    "DepthInjection":               DepthInjection,
    "DepthRippleEffect":            DepthRippleEffect,
    "DepthBlender":                 DepthBlender,
    "DepthShapeModifier":           DepthShapeModifier,

    "DepthShapeModifierPrecise":    DepthShapeModifierPrecise,
    # "DepthMapProtrusion":          DepthMapProtrusion,
    #feature modulation
    "FeatureMixer":                 FeatureMixer,
    "FeatureCombine":               FeatureCombine,
    "FeatureOscillator":            FeatureOscillator,
    "FeatureScaler":                FeatureScaler,
    "FeatureSmoothing":             FeatureSmoothing,
    "FeatureFade":                  FeatureFade,
    "FeatureMath":                  FeatureMath,
    "PreviewFeature":               PreviewFeature,
    "AnimatedFeaturePreview":       AnimatedFeaturePreview,
    "FeatureRebase":                FeatureRebase,
    "FeatureTruncateOrExtend":      FeatureTruncateOrExtend,
    "FeatureAccumulate":            FeatureAccumulate,
    "FeatureContiguousInterpolate": FeatureContiguousInterpolate,
    "FeatureRenormalize":           FeatureRenormalize,
    "FeatureToFlexIntParam":        FeatureToFlexIntParam,
    "FeatureToFlexFloatParam":      FeatureToFlexFloatParam,
    
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
    "FlexImageWarp":                FlexImageWarp,
    "FlexImageVignette":            FlexImageVignette,
    "FlexImageTransform":           FlexImageTransform,
    "FlexImageHueShift":            FlexImageHueShift,
    "FlexImageDepthWarp":           FlexImageDepthWarp,
    "FlexImageHorizontalToVertical":FlexImageHorizontalToVertical,



    #visulizers
    "ProximityVisualizer":          ProximityVisualizer,
    "EffectVisualizer":             EffectVisualizer,
    "PitchVisualizer":              PitchVisualizer,

    #garb   
    "DyeImage":                     DyeImage,
    "Color_Picker":                  Color_Picker,
    "ImageCASBatch":                ImageCASBatch,
    "ImageScaleToTarget":           ImageScaleToTarget,
    "MovingShape":                  MovingShape,
    "_mfc":                         _mfc,
    "TextMaskNode":                 TextMaskNode,
    

    #TODO: make useful
    # "MaskCompositePlus":                MaskCompositePlus,
    
    "AdvancedLuminanceMask":        AdvancedLuminanceMask,
    "TranslucentComposite":         TranslucentComposite,

    #utility nodes
    "ImageChunk":                   ImageChunks, 
    "ImageInterval":                ImageIntervalSelect,
    "VideoChunk":                   VideoChunks,
    "ImageDifference":              ImageDifference,
    "Image_Shuffle":                Image_Shuffle,
    "SwapDevice":                   SwapDevice,
    "ImageIntervalSelectPercentage":ImageIntervalSelectPercentage,
    "ImageIndexSelect":             ImageIndexSelect,
    
    "MIDIToAudio": MIDIToAudio,
    "MIDILoader": MIDILoader,
}


WEB_DIRECTORY = "./web/js"
EXTENSION_WEB_DIRS = ["./web/extensions"]

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSeparatorSimple":         "Audio Separator",
    "ProximityVisualizer":          "Preview Proximity",
    "EffectVisualizer":             "Preview FeatureEffect",
    "PitchVisualizer":              "Preview Pitch",
    "FlexVideoSpeed":               "**BETA** Flex Video Speed",
    "FlexVideoFrameBlend":          "**BETA**Flex Video Frame Blend",
    "AudioFeatureVisualizer":       "Audio Feature Visualizer ***BETA***",

    "MIDILoadAndExtract":           "MIDI Load & Feature Extract",
    "PitchRangeByNoteNode":         "Pitch Range By Note",
    "AudioFeatureExtractor":        "Audio Feature Extractor",
    "TimeFeatureNode":              "Time Feature",
    "DepthFeatureNode":             "Depth Feature",
    "BrightnessFeatureNode":        "Brightness Feature",
    "MotionFeatureNode":            "Motion Feature",
    "ImageCASBatch":                "Image Contrast Adaptive Sharpen Batch",
    "ImageIntervalSelectPercentage":"Image Interval Select %",
    "ImageScaleToTarget":           "Upscale Image To Target",
    "FeatureToSplineData":          "***BETA*** Feature To Spline Data",
    "SplineFeatureModulator":       "***BETA*** Spline Feature Modulator",
    "SplineRhythmModulator":        "***BETA*** Spline Rhythm Modulator",
    "MaskMorph":                    "Mask Morph [DEPRECATED]",
    "MaskTransform":                "Mask Transform [DEPRECATED]",
    "MaskMath":                    "Mask Math [DEPRECATED]",
    "MaskRings":                   "Mask Rings [DEPRECATED]",
    "MaskWarp":                    "Mask Warp [DEPRECATED]",

    "Image_Shuffle":                 "Image Shuffle ROTI",
    "Audio_Combine":                 "Audio Combine ROTI",
    "Audio_Concatenate":             "Audio Concatenate ROTI",
    "Color_Picker":                  "Color_Picker ROTI",
    "Doom_":                        "DOOM",
}



from .nodes.acestep.nodes import NODE_CLASS_MAPPINGS as ACEStepNODE_CLASS_MAPPINGS
from .nodes.acestep.nodes import NODE_DISPLAY_NAME_MAPPINGS as ACEStepNODE_DISPLAY_NAME_MAPPINGS

# Update NODE_CLASS_MAPPINGS with external nodes
NODE_CLASS_MAPPINGS.update(EXTERNAL_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ACEStepNODE_CLASS_MAPPINGS)  
NODE_DISPLAY_NAME_MAPPINGS.update(ACEStepNODE_DISPLAY_NAME_MAPPINGS)


import re

suffix = " ⚡🅡🅞🅣🅘"

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

# if hasattr(PromptServer, "instance"):
#     # NOTE: we add an extra static path to avoid comfy mechanism
#     # that loads every script in web. 
#     # 
#     # Again credit to KJNodes and MTB nodes
#     PromptServer.instance.app.add_routes(
#         [web.static("/ryanontheinside_web_async", (Path(__file__).parent.absolute() / "ryanontheinside_web_async").as_posix())]
#     )
# #register tooltips after all classes are initialized
if not DISABLE_TOOLTIPS:
    register_all_tooltips()
