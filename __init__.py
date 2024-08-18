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
)
from .nodes.masks.audio_masks import (
    AudioMaskMorph,
    AudioMaskWarp,
    AudioMaskTransform,
    AudioMaskMath
)
from .nodes.masks.optical_flow_masks import (
    OpticalFlowMaskModulation,
    OpticalFlowDirectionMask,
    OpticalFlowParticleSystem
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

from .nodes.masks.utility_nodes import _mfc, TextMaskNode, MovingShape

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


    "AudioMaskMorph":AudioMaskMorph,
    "AudioMaskWarp": AudioMaskWarp,
    "AudioMaskTransform":AudioMaskTransform,
    "AudioMaskMath": AudioMaskMath,
    "AudioSeparator": AudioSeparator,
    "AudioFeatureVisualizer":AudioFeatureVisualizer,


    "MovingShape": MovingShape,
    "_mfc":_mfc,
    "TextMaskNode":TextMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskMorph": "Temporal Mask Morph | RyanOnTheInside",
    "MaskTransform":"Temporal Mask Transform | RyanOnTheInside",
    "MaskMath":"Temporal Mask Math | RyanOnTheInside",
    "MaskRings":"Temporal Mask Rings | RyanOnTheInside",
    "MaskWarp":"Mask Warp | RyanOnTheInside",

    
    "OpticalFlowMaskModulation": "Optical Flow Mask Modulation | RyanOnTheInside",
    "OpticalFlowParticleSystem":"Optical Flow Particle System | RyanOnTheInside",
    #"OpticalFlowDirectionMask":"Optical Flow Direction Mask | RyanOnTheInside",
    
    "ParticleEmissionMask":"Particle Emission Mask | RyanOnTheInside",
    "Vortex": "Vortex | RyanOnTheInside",
    "GravityWell":"Gravity Well | RyanOnTheInside",
    "ParticleEmitter": "Particle Emitter | RyanOnTheInside",
    "EmitterMovement":"Emitter Movement | RyanOnTheInside",
    "SpringJointSetting":"Spring Joint Setting | RyanOnTheInside",
    "StaticBody":"Static Body | RyanOnTheInside",
    "ParticleColorModulation":"Particle Color Modulation | RyanOnTheInside",
    "ParticleSizeModulation": "Particle Size Modulation | RyanOnTheInside",
    "ParticleSpeedModulation":"Particle Speed Modulation | RyanOnTheInside",

    "AudioMaskMorph":"Audio Mask Morph | RyanOnTheInside",
    "AudioMaskWarp": "Audio Mask Warp | RyanOnTheInside",
    "AudioMaskTransform":"Audio Mask Transform | RyanOnTheInside",
    "AudioMaskMath": "Audio Mask Math | RyanOnTheInside",
    "AudioSeparator": "Audio Separator | RyanOnTheInside",
    "AudioFeatureVisualizer": "Audio Feature Visualizer | RyanOnTheInside" ,

    "MovingShape": "Moving Shape | RyanOnTheInside",
    "TextMaskNode":"Text Mask Node | RyanOnTheInside"
}