# ACEStep Native - ComfyUI Repaint & Extend

Native ComfyUI implementation of ACEStep audio repaint and extend functionality. This nodepack provides seamless integration with ComfyUI's existing sampling infrastructure to enable selective audio regeneration and extension without requiring custom types.

## Features

- **🎨 Repaint**: Selectively regenerate specific time ranges in existing audio
- **📏 Extend**: Add new content before or after existing audio  
- **🔄 Hybrid**: Combine repaint and extend operations simultaneously
- **🎯 Native Integration**: Works with ComfyUI's SamplerCustomAdvanced
- **⚡ Time-based Controls**: Intuitive time-based parameter interface
- **🔧 Debug Tools**: Visualization and analysis utilities

## Installation

1. Clone or download this repository to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone <repository-url> acestep_native
```

2. Restart ComfyUI - the nodes will automatically be available.

## Node Overview

### Core Functionality Nodes

#### ACEStep Repaint Guider
Creates a guider for repainting specific audio regions.
- **Inputs**: Model, conditioning, source latents, time range, strength settings
- **Output**: GUIDER (for use with SamplerCustomAdvanced)

#### ACEStep Extend Guider  
Creates a guider for extending audio before/after existing content.
- **Inputs**: Model, conditioning, source latents, extend times
- **Output**: GUIDER (for use with SamplerCustomAdvanced)

#### ACEStep Hybrid Guider
Combines repaint and extend functionality in a single node.
- **Inputs**: All repaint/extend parameters with optional controls
- **Output**: GUIDER (for use with SamplerCustomAdvanced)

### Utility Nodes

#### ACEStep Analyze Latent
Analyzes ACE latent properties for debugging.

#### ACEStep Time Range
Converts time ranges to frame indices.

#### ACEStep Mask Visualizer
Visualizes repaint masks for debugging.

## Basic Workflows

### Repaint Workflow

**Scenario**: Replace audio content from 10-20 seconds with new generation.

```
[LoadAudio] → [VAEEncodeAudio] → [source_latents]
                                      ↓
[TextEncodeAceStepAudio] → [positive] ↓
[TextEncodeAceStepAudio] → [negative] ↓
[CheckpointLoaderSimple] → [model]    ↓
                                      ↓
[ACEStepRepaintGuider] ← [start_time: 10.0]
                        ← [end_time: 20.0] 
                        ← [repaint_strength: 0.7]
                        ← [feather_time: 0.1]
                        ↓
[RandomNoise] → [SamplerCustomAdvanced] ← [BasicScheduler]
                        ↓              ← [KSamplerSelect]
[VAEDecodeAudio] ← [output latent]
                        ↓
                   [SaveAudio]
```

### Extend Workflow

**Scenario**: Add 10 seconds of new content after existing audio.

```
[LoadAudio] → [VAEEncodeAudio] → [source_latents]
                                      ↓
[TextEncodeAceStepAudio] → [positive] ↓
[TextEncodeAceStepAudio] → [negative] ↓  
[CheckpointLoaderSimple] → [model]    ↓
                                      ↓
[ACEStepExtendGuider] ← [extend_left_time: 0.0]
                      ← [extend_right_time: 10.0]
                      ↓
[RandomNoise] → [SamplerCustomAdvanced] ← [BasicScheduler]
                      ↓                ← [KSamplerSelect] 
[VAEDecodeAudio] ← [output]
                      ↓
                 [SaveAudio]
```

### Hybrid Workflow

**Scenario**: Extend audio by 5 seconds on each side AND repaint middle section.

```
[LoadAudio] → [VAEEncodeAudio] → [source_latents]
                                      ↓
[TextEncodeAceStepAudio] → [positive] ↓
[TextEncodeAceStepAudio] → [negative] ↓
[CheckpointLoaderSimple] → [model]    ↓
                                      ↓
[ACEStepHybridGuider] ← [extend_left_time: 5.0]
                      ← [extend_right_time: 5.0]
                      ← [repaint_start_time: 15.0]  
                      ← [repaint_end_time: 25.0]
                      ← [repaint_strength: 0.8]
                      ↓
[RandomNoise] → [SamplerCustomAdvanced] ← [BasicScheduler]
                      ↓                ← [KSamplerSelect]
[VAEDecodeAudio] ← [output]
                      ↓
                 [SaveAudio]
```

## Parameter Guide

### Time Parameters
- **Times are in seconds** (e.g., 10.5 = 10.5 seconds)
- **Frame conversion**: ~10.8 frames per second for ACE audio latents
- **Precision**: 0.1 second minimum increment

### Repaint Parameters
- **start_time/end_time**: Define the region to repaint
- **repaint_strength**: 0.0 = no change, 1.0 = complete replacement
- **feather_time**: Smooth transition duration at edges

### Extend Parameters  
- **extend_left_time**: Seconds to add before audio
- **extend_right_time**: Seconds to add after audio
- Set to 0.0 to disable extension on that side

## Technical Details

### ACE Audio Latent Format
- **Shape**: (batch, 8, 16, length)
- **Channels**: 8 (ACE-specific encoding)
- **Height**: 16 (fixed for ACE)
- **Length**: Variable (time dimension)
- **Frame Rate**: ~10.8 frames/second

### Time to Frame Conversion
```python
frame_index = int(time_seconds * 44100 / 512 / 8)
time_seconds = frame_index * 512 * 8 / 44100
```

### Masking Strategy
- **Repaint**: Binary mask with optional feathering for smooth transitions
- **Extend**: Zero padding with masks for extension regions
- **Blending**: Timestep-aware blending preserves original content

### Extend Architecture
**ACEStep extend operations are handled entirely within the guider:**

**ACEStepExtendGuider** handles extension internally by:
- Detecting when the input latent needs extension
- Creating extended latent with original content positioned correctly  
- Generating appropriate noise for the extended dimensions
- Preserving original audio content during sampling
- Allowing generation only in the extended regions

This unified approach eliminates the need for separate latent preparation steps.

## Troubleshooting

### Common Issues

**"source_latents must be audio latents"**
- Ensure you're using VAEEncodeAudio output, not image latents
- Check that latent type is "audio"

**"start_time must be less than end_time"**  
- Verify time parameters are in correct order
- Both times must be positive

**"At least one extend time must be > 0"**
- For extend operations, set either left or right extend time > 0

### Performance Tips

1. **Use feathering** (0.1-0.2s) for smooth transitions in repaint
2. **Start with lower repaint_strength** (0.5-0.7) and adjust up
3. **Limit extend times** to reasonable durations to manage memory
4. **Use debug nodes** to verify mask patterns before full generation

### Quality Tips

1. **Match conditioning** between original and repaint regions
2. **Use appropriate CFG values** (6-12 typically work well)
3. **Consider feather_time** based on audio content (longer for musical, shorter for speech)
4. **Test time ranges** with ACEStepTimeRange node first

## Advanced Usage

### Custom Noise Patterns
Use RandomNoise with different seeds for extend regions to create variety.

### Conditional Repainting
Adjust repaint_strength based on how much change you want:
- 0.3-0.5: Subtle modifications
- 0.6-0.8: Moderate changes  
- 0.9-1.0: Complete replacement

### Multiple Operations
Chain operations by using output of one as input to another for complex edits.

## Compatibility

- **ComfyUI**: Latest version with SamplerCustomAdvanced support
- **ACEStep Models**: All standard ACE checkpoints
- **Audio Formats**: Works with standard audio VAE encode/decode
- **Dependencies**: None beyond standard ComfyUI

## Contributing

Issues and pull requests welcome! This implementation follows ComfyUI's patterns and should integrate seamlessly with existing workflows.

## License

Same as ComfyUI - GPL-3.0

---

*This nodepack implements the functionality from the community ace-step nodepack using native ComfyUI infrastructure for better compatibility and maintainability.*

## Audio Fidelity Preservation

### The VAE Round-Trip Problem

Even with perfect repaint/extend logic, the original audio will lose some fidelity due to:
- **Encoding loss**: Original audio → latent space (lossy compression)
- **Decoding loss**: Latent space → reconstructed audio (additional artifacts)
- **VAE imperfections**: The audio VAE isn't perfect and introduces subtle changes

This means that even "preserved" regions will sound slightly different from the original.

### Solution: ACEStepAudioPostProcessor

The `ACEStepAudioPostProcessor` node solves this by:
1. Taking the original audio (before VAE processing)
2. Taking the processed audio (after VAE processing)
3. Using the guider's mask information to splice back the original audio in preserved regions
4. Applying smooth crossfading at boundaries to avoid audio artifacts

### Example Workflow

```
[Load Audio] → [VAE Encode Audio] → [ACEStep Repaint Guider] → [SamplerCustomAdvanced] → [VAE Decode Audio]
     ↓                                        ↓                                                    ↓
[Original Audio] ────────────────────────────[Guider]──────────────────────────────────→ [ACEStep Audio Post Processor]
                                                                                                   ↓
                                                                                          [Final High-Fidelity Audio]
```

### Benefits

- **Perfect preservation**: Original audio regions maintain 100% fidelity
- **Seamless transitions**: Crossfading prevents clicks and pops at boundaries
- **Automatic masking**: Uses the same mask logic as the guiders
- **Easy to use**: Just connect the original audio, processed audio, and guider

### Parameters

- **crossfade_duration**: Duration in seconds for smooth transitions at mask boundaries (default: 0.1s)
  - 0.0 = hard cuts (may cause clicks)
  - 0.1-0.2 = smooth transitions (recommended)
  - Higher values = more gradual blending 