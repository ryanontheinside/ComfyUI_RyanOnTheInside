from .nodes.masks.temporal_masks import (
    MaskMorph,
    MaskTransform,
    MaskMath,
    MaskRings,
    MaskWarp,
    #, MaskBlur, MaskThreshold, MaskTransform
    ) 
from .nodes.masks.audio_masks import AudioControlledMaskMorph

from .nodes.masks.optical_flow_masks import (
    OpticalFlowMaskModulation,
    OpticalFlowDirectionMask,
    OpticalFlowParticleSystem
    )



from .nodes.masks.particle_system_masks import (
    ParticleEmissionMask,
    #PulsatingParticleSystemMask,
    Vortex,
    GravityWell,
    ParticleEmitter,
    EmitterMovement,
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
    #"PulsatingParticleSystemMask":PulsatingParticleSystemMask,
    "Vortex":Vortex,
    "GravityWell":GravityWell,
    "EmitterMovement":EmitterMovement,
    "ParticleEmitter":ParticleEmitter,
   # "AudioControlledMaskMorph": AudioControlledMaskMorph,
    
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
    
    # "PulsatingParticleSystemMask":"Pulsating Particle System Mask | RyanOnTheInside",
    "ParticleEmissionMask":"Particle Emission Mask | RyanOnTheInside",
    "Vortex": "Vortex | RyanOnTheInside",
    "GravityWell":"Gravity Well | RyanOnTheInside",
    "ParticleEmitter": "Particle Emitter | RyanOnTheInside",
    "EmitterMovement":"Emitter Movement | RyanOnTheInside",
   # "AudioControlledMaskMorph": "Audio Controlled Mask Morph | RyanOnTheInside",
    
    "MovingShape": "Moving Shape | RyanOnTheInside",
    "TextMaskNode":"Text Mask Node | RyanOnTheInside"
}