# ACE-Step 1.5 Latent Channel Semantics

## Overview

This document presents findings from a systematic investigation of the 64 latent channels in ACE-Step 1.5's audio VAE. The goal was to determine what each channel (or group of channels) controls in the decoded audio output, enabling better-informed creative tools like the `ACEStep15LatentChannelEQ` node.

**Key finding**: The 64 channels are NOT organized as simple frequency bands. Instead, they encode a mix of spectral balance, timbral texture, and harmonic structure, with a few "keystone" channels that dramatically reshape the output when perturbed. The current 4-band EQ grouping (16 channels each) is suboptimal and should be restructured.

## Methodology

### Experiment Design

All experiments perturb the **same latent tensor** (deterministic sampling with seed=31) to isolate the VAE decoder's channel-to-audio mapping. By scaling specific channels between KSampler and VAEDecodeAudio, we observe what the VAE maps each channel to without diffusion confounds.

- **Model**: ace_step_1.5_turbo_aio.safetensors (turbo checkpoint)
- **Sampling**: KSampler, euler, simple scheduler, 8 steps, CFG 1.0, denoise 1.0
- **ModelSamplingAuraFlow** shift=3 (turbo model)
- **Prompt**: Neo-soul track with live instrumentation (guitar, drums, vocals)
- **Duration**: 30 seconds
- **Seed**: 31 (fixed)
- **Output**: 48kHz FLAC audio

### Perturbation Types

1. **Group perturbation** (8 groups of 8 channels): Zero (0x), attenuate (0.5x), boost (1.5x)
2. **Individual channel zeroing** (all 64 channels): Zero out one channel at a time

### Analysis Metrics

For each output audio, computed:
- **RMS energy** (overall loudness)
- **Spectral centroid** (average frequency / brightness)
- **Frequency band energies**: sub-bass (20-60Hz), bass (60-250Hz), low-mid (250-500Hz), mid (500-2kHz), upper-mid (2-4kHz), presence (4-6kHz), brilliance (6-20kHz)
- **Onset strength** (rhythmic/transient content)
- **Zero crossing rate** (noisiness/percussiveness indicator)
- **MFCC distance** (timbral shift from baseline)
- **Chroma distance** (harmonic/pitch content shift)

All metrics reported as percentage change from baseline.

## Baseline Characteristics

| Metric | Value |
|--------|-------|
| Sample rate | 48,000 Hz |
| Duration | 30.0s |
| RMS | 0.185 |
| Peak | 1.000 |
| Spectral centroid | 2,565 Hz |
| Onset strength (mean) | 1.226 |
| ZCR (mean) | 0.042 |

## Group-Level Findings

### Frequency Band Impact Table (% change when group is zeroed)

| Group | Channels | Sub-bass | Bass | Low-mid | Mid | Upper-mid | Presence | Brilliance |
|-------|----------|----------|------|---------|-----|-----------|----------|------------|
| G0 | 0-7 | -16.0 | **-23.9** | -16.1 | -0.9 | -14.3 | -2.3 | +0.9 |
| G1 | 8-15 | **-34.7** | **-33.8** | **-31.9** | -16.3 | **-31.0** | **-34.0** | -5.1 |
| G2 | 16-23 | **-40.0** | +1.4 | +16.8 | **-43.6** | **-55.9** | **-53.5** | -19.4 |
| G3 | 24-31 | -6.5 | +8.5 | **+43.3** | **+48.4** | +33.7 | +6.9 | -11.6 |
| G4 | 32-39 | **-25.0** | **-20.5** | -9.7 | -9.3 | +3.9 | -11.3 | +2.6 |
| G5 | 40-47 | -18.7 | -16.2 | -7.6 | -14.0 | +5.3 | -11.3 | -5.2 |
| G6 | 48-55 | **-53.8** | **-23.1** | -12.1 | -12.3 | -1.6 | -22.9 | -4.8 |
| G7 | 56-63 | +7.1 | +9.2 | +17.2 | +5.2 | -16.3 | -22.5 | **+41.6** |

### Group Semantic Profiles

#### G0 (channels 0-7): Low-Frequency Foundation
- **Dominant effect**: Bass removal (-24% bass, -16% sub-bass, -16% low-mid)
- **Character**: Broad low-frequency energy. Zeroing removes warmth/body without significantly affecting mids or highs.
- **MFCC distance**: 19.3 (moderate timbral shift)
- **Behavior**: Symmetric — boosting 1.5x adds proportional bass energy (+14% bass, +10% sub-bass)

#### G1 (channels 8-15): Broadband Energy / Master Volume
- **Dominant effect**: Uniform broadband reduction (-34% sub-bass through -34% presence)
- **Character**: Most impactful group for overall loudness (-18% RMS when zeroed). Acts like a master volume/energy control affecting all frequency bands roughly equally.
- **MFCC distance**: 62.2 (large timbral shift)
- **ZCR change**: +35% when zeroed — remaining signal becomes noisier/thinner
- **Notable**: Contains channel 13 (4th most impactful individual channel) and channel 14 (5th most impactful)

#### G2 (channels 16-23): Mid/Upper-Mid Texture & Presence
- **Dominant effect**: Massive upper-mid/presence removal (-56% upper-mid, -54% presence, -44% mid)
- **Character**: Controls the "forward" quality of the mix — vocal presence, instrument definition, clarity. Sub-bass also drops -40% suggesting coupling.
- **MFCC distance**: 109.6 (LARGEST timbral shift of any group)
- **Notable**: Contains channel 19 (THE most impactful individual channel, impact=107)

#### G3 (channels 24-31): Spectral Balance / Brightness Control
- **Dominant effect**: INVERSE behavior — zeroing INCREASES mid energy (+48% mid, +43% low-mid)
- **Character**: Acts as a high-frequency counterweight. Removing it shifts energy downward (centroid drops -27%). Functions like a brightness/presence balance control.
- **MFCC distance**: 82.4 (very large timbral shift)
- **Notable**: Contains channel 29 (2nd most impactful, impact=99) which controls spectral tilt

#### G4 (channels 32-39): Sub-bass & Low Foundation
- **Dominant effect**: Sub-bass/bass removal (-25% sub-bass, -21% bass)
- **Character**: Similar to G0 but more focused on sub-bass. Moderate overall impact.
- **MFCC distance**: 7.3 (smallest timbral shift — mostly energy redistribution, not timbre change)

#### G5 (channels 40-47): Broadband Body
- **Dominant effect**: Moderate broadband reduction, slightly sub-bass weighted (-19% sub-bass, -16% bass, -14% mid)
- **Character**: General body/fullness. Less impactful than G1 but similar broadband character.
- **MFCC distance**: 12.2 (moderate)

#### G6 (channels 48-55): Sub-bass Dominance
- **Dominant effect**: Strongest sub-bass impact of any group (-54% sub-bass, -23% bass)
- **Character**: Primary sub-bass energy channel. Zeroing dramatically removes low-end weight.
- **MFCC distance**: 16.5 (moderate — focused on bass, not timbre)
- **RMS change**: -17% (significant loudness drop, since sub-bass carries lots of energy)

#### G7 (channels 56-63): High-Frequency Air & Brilliance
- **Dominant effect**: Brilliance boost when zeroed (+42%), centroid rises +19%
- **Character**: INVERSE behavior like G3 — these channels suppress high-frequency content. Removing them releases brilliance/air but reduces presence (-22%). Acts as a high-frequency damper.
- **MFCC distance**: 69.1 (large timbral shift — changes the "sheen" of the audio)
- **Notable**: Contains channel 56 (3rd most impactful, impact=91)

## Individual Keystone Channels

Six channels have outsized impact when zeroed (composite impact score > 20):

### Channel 19 (Impact: 107) — "Presence/Definition Control"
- **RMS**: +5.9% (louder when removed)
- **Spectral centroid**: +11.4% (brighter)
- **Upper-mid**: -67.5%, Presence: -61.7%, Mid: -35.4%
- **Brilliance**: -43.4%, but ZCR: +32.1% (noisier)
- **MFCC distance**: 88.5 (massive timbre change)
- **Interpretation**: This channel encodes the core "body" of mid-frequency content. Removing it strips definition and presence while making the residual signal noisier and brighter (because the remaining energy skews high). This is the single most important channel for timbral character.

### Channel 29 (Impact: 99) — "Spectral Tilt / Low-Shift"
- **RMS**: +4.7%
- **Spectral centroid**: -26.1% (much darker)
- **Mid**: +24.0%, Bass: +9.3%, Sub-bass: +5.1%
- **Brilliance**: -23.9%, ZCR: -13.2%
- **MFCC distance**: 67.5
- **Interpretation**: Encodes high-frequency spectral energy. Removing it causes a dramatic downward spectral shift — everything gets darker and more bass-heavy. Works as a spectral tilt control.

### Channel 56 (Impact: 91) — "Air & Shimmer"
- **RMS**: +4.7%
- **Spectral centroid**: +18.9% (brighter)
- **Brilliance**: +25.0%, but Upper-mid: -26.8%, Presence: -29.5%
- **ZCR**: +18.0% (noisier)
- **MFCC distance**: 62.7
- **Interpretation**: Encodes presence-range energy (4-6kHz). Removing it shifts brightness upward (more air/shimmer) but removes the defined presence band. Related to "breathiness" and high-frequency texture.

### Channel 13 (Impact: 74) — "High-Frequency Texture"
- **RMS**: -4.2%
- **Spectral centroid**: +16.5% (brighter)
- **Brilliance**: +26.6%, ZCR: +37.8%
- **Presence**: -10.4%, Bass: -10.7%
- **MFCC distance**: 49.6
- **Interpretation**: Encodes mid-range body. Removing it makes the output brighter and noisier, with significantly increased ZCR indicating more high-frequency transient content.

### Channel 14 (Impact: 32) — "Full Spectrum Body"
- **RMS**: -7.5%
- **All bands negative**: Sub-bass -17%, Bass -12%, Mid -12%, Upper-mid -29%, Presence -24%, Brilliance -30%
- **MFCC distance**: 18.6
- **Interpretation**: Broad energy carrier — removing it uniformly reduces content across all bands, like a volume knob that also shifts timbre.

### Channel 23 (Impact: 27) — "Low-Mid Warmth"
- **RMS**: -8.7%
- **Spectral centroid**: -9.4% (darker)
- **Sub-bass**: -16%, Bass: -17%, Mid: -19%
- **But Upper-mid**: +16%, Presence: +14%
- **ZCR**: -15.6%
- **MFCC distance**: 8.1
- **Interpretation**: Encodes low-mid warmth. Removing it loses low-end body but reveals more upper-mid/presence — acts like a warmth/clarity balance within the low-mid range.

## Channel Clustering & Proposed Semantic Groupings

Based on the analysis, the channels naturally cluster into 6 functional categories rather than the current 4 arbitrary bands:

### Implemented 6-Band Structure (Phase 1+2 Validated)

| Band | Channels | Label | Description |
|------|----------|-------|-------------|
| **Bass** | 0-7, 32-39 (G0+G4) | Low-Frequency Energy | Both reduce bass/volume when attenuated. Near-linear response. |
| **Brightness** | 8-15 (G1 only) | Spectral Brightness | Massive centroid control (+21%). INVERTED — attenuating brightens. Separated from G5. |
| **Body** | 40-55 (G5+G6) | Broadband Fullness | Gentle broadband attenuators. G6 weak but directionally consistent with G5. |
| **Texture** | 16-23 (G2) | Timbral Character | Mid-range MFCC impact. EMERGENT — unpredictable across genres. Contains keystone ch19. |
| **Tilt** | 24-31 (G3) | Spectral Tilt | Clean spectral tilt. Model compensates, so pushed harder. Contains keystone ch29. |
| **Air** | 56-63 (G7) | Energy & Shimmer | Broadband energy + brilliance + onset activity. INVERTED. Contains keystone ch56. |

### Why the Current 4-Band Split is Wrong

The current EQ splits channels into 4 equal bands of 16 (0-15, 16-31, 32-47, 48-63). This grouping has several problems:

1. **Band 1 (0-15) mixes two different functions**: G0 (bass foundation) and G1 (broadband energy) have fundamentally different roles. Adjusting this band simultaneously changes low-end and overall volume.

2. **Band 2 (16-31) mixes opposing behaviors**: G2 (removes presence when zeroed) and G3 (ADDS mid energy when zeroed) are spectral opposites. Scaling them together partially cancels their effects.

3. **Band 3 (32-47) is relatively benign**: G4 and G5 have similar, moderate, mostly bass-oriented effects. This band works OK as-is but has limited creative utility.

4. **Band 4 (48-63) mixes opposing behaviors**: G6 (sub-bass removal) and G7 (brilliance release) are also opposing — one affects lows, the other highs.

## Future Investigation: Higher-Level Musical Dimensions

Phase 2 found guidance-specific effects that go beyond spectral/timbral changes:

- **Rhythm/onset changes**: G7 (air) guidance increased onset strength +11%. ch14 (body) guidance decreased it -4 to -8%. The model generates different rhythmic patterns, not just different frequency balances.
- **Harmonic content changes**: ch14 had high chroma distance (0.10-0.12), indicating actual pitch/harmonic content shifts — potentially different notes, voicings, or chord structures.

These effects suggest latent channel guidance can control *what* the model produces (instrument density, note patterns, rhythmic complexity) — not just *how it sounds*. However, Phase 1/2 metrics were all audio-signal level (RMS, centroid, MFCC, onset strength). To map higher-level musical dimensions, a follow-up study should:

- Use perceptual listening analysis across multiple seeds/prompts/genres
- Apply music-level metrics: note density, instrument count (via source separation), chord complexity, melodic contour
- Test wider perturbation ranges and channel combinations
- Investigate whether specific channel combinations can reliably control instrument count, melodic variation, or structural complexity

This could unlock controls like "more synth layers," "simpler arrangement," or "more melodic variation" — creative tools that go far beyond EQ.

## Current Node Design: Generation Steering + EQ + Keystone Config

*Redesigned based on Phase 1+2 multi-seed statistical validation (10 seeds × 6 genres + 5 seeds × 4 genres).*

Three nodes implement the latent channel control system:

1. **ACE-Step 1.5 Generation Steering** — The primary creative tool. Runs the model twice per step (guidance-based). Changes what the model *generates*, not just how it sounds. Sensitivity-normalized for the guidance system.
2. **ACE-Step 1.5 Latent Channel EQ** — Multiplicative scaling at various pipeline points. Equivalent to post-processing EQ. Simpler, less powerful, but useful for fine-tuning.
3. **ACE-Step 1.5 Keystone Config** — Optional config node providing per-channel gains for 5 high-impact channels. Connects to either node's `keystone_config` input.
4. **ACE-Step 1.5 Musical Controls** — High-level musical properties (rhythm, harmony, dynamics). *Needs Phase 3 re-validation with new groupings.*

Both Generation Steering and EQ share the same 6-band structure, sensitivity factors, and keystone support via a shared mixin (`_LatentChannelBandMixin`).

### 6-Band Structure

| Band | Input | Channels | Sensitivity | Description |
|------|-------|----------|-------------|-------------|
| Bass | `bass_gain` | 0-7, 32-39 (G0+G4) | +0.9 | Low-frequency energy and overall volume. Both groups reduce bass/volume when attenuated. |
| Brightness | `brightness_gain` | 8-15 (G1 only) | **-0.7** (INVERTED) | Spectral brightness. Strongest spectral control (+21% centroid at gs=1). Boosting the slider attenuates G1, which brightens output. |
| Body | `body_gain` | 40-55 (G5+G6) | +1.0 | Broadband fullness. Both are gentle broadband attenuators. |
| Texture | `texture_gain` | 16-23 (G2) | +0.5 | Mid-range timbral character (MFCC +27 at gs=1). EMERGENT — unpredictable, effects vary by genre. Conservative sensitivity. |
| Tilt | `tilt_gain` | 24-31 (G3) | +1.5 | Spectral balance. Clean spectral tilt (centroid -9.5% at gs=1). Model compensates, so pushed harder. |
| Air | `air_gain` | 56-63 (G7) | **-1.0** (INVERTED) | High-frequency shimmer, energy, and rhythmic activity. Boosting the slider attenuates G7, which adds energy/shimmer. |

**Key design changes from previous version:**
- **Brightness separated from Body**: G1 was previously grouped with G5 as "body". G1 is a massive spectral brightener (+21% centroid), while G5 is a gentle volume knob. Totally different behaviors.
- **Inverse channels handled**: Brightness (G1) and Air (G7) have negative sensitivity — the UI now works intuitively ("brightness UP" = brighter output).
- **Body redefined**: Now G5+G6 (gentle broadband attenuators) instead of G1+G5 (mismatched behaviors).

### Keystone Config Node

The `ACEStep15KeystoneConfig` node outputs a `KEYSTONE_CONFIG` dict that connects to either node's optional `keystone_config` input. It provides individual gain controls for 5 high-impact channels (redesigned based on multi-seed statistical significance):

| Input | Channel | Sensitivity | Inverted? | Effect |
|-------|---------|-------------|-----------|--------|
| `presence` | 19 | +1.8 | No | +8% RMS, +3% centroid. Late-stage dominant. Most impactful single channel. |
| `spectral_tilt` | 29 | +1.2 | No | -13% centroid, +8% RMS. Late-stage. Shifts spectrum brighter/darker. |
| `energy` | 56 | **-1.0** | **Yes** | +13% RMS, +7% centroid, +3% onset. Boosting attenuates ch56 → more energy/shimmer. |
| `brilliance` | 13 | **-0.7** | **Yes** | +18.5% centroid. Boosting attenuates ch13 → more brilliance. Late-stage dominant. |
| `weight` | 2 | +1.0 | No | -4.3% RMS, spectrally neutral. Late-stage bass weight control. |

**Changes from previous version:**
- **Dropped ch14 ("body") and ch23 ("warmth")**: Not statistically significant across the multi-seed grid.
- **Added ch2 ("weight")**: Statistically significant bass weight control, replacing ch14/ch23.
- **Inverse channels (ch56, ch13) now have negative sensitivity**: UI works intuitively — "energy UP" = more energy.

Keystone gains multiply on top of band gains — they refine individual channels within their parent band.

### Internal Sensitivity Normalization

All sliders use a uniform 0-2 range with 1.0 as neutral. Internally, each band and keystone has a sensitivity factor derived from Phase 2 guidance testing across multiple seeds and genres. The formula is: `internal_gain = 1.0 + sensitivity * (user_gain - 1.0)`.

**Negative sensitivity** means the internal gain moves opposite to user input. For example, brightness sensitivity = -0.7: user sets slider to 1.5 → internal_gain = 1.0 + (-0.7) × 0.5 = 0.65 → G1 channels attenuated → brighter output.

### Inverse-Behavior Awareness

The Brightness band (ch 8-15) and Air band (ch 56-63) exhibit **inverse behavior** — attenuating them INCREASES their named quality (brightness/air). This is now handled automatically via negative sensitivity factors, so the UI works intuitively.

## Raw Data Summary

### Group-Level Impact Scores (composite of |RMS%| + |centroid%| + |onset%| + MFCC_distance)

| Group | Zero Impact | Half Impact | Boost Impact | Primary Effect |
|-------|------------|-------------|--------------|----------------|
| G0 (0-7) | 31.6 | 15.1 | 18.5 | Bass foundation |
| G1 (8-15) | 95.4 | 53.1 | 51.1 | Broadband energy |
| G2 (16-23) | 124.3 | 64.7 | 60.5 | Mid/presence texture |
| G3 (24-31) | 114.4 | 64.4 | 71.7 | Spectral balance |
| G4 (32-39) | 21.1 | 10.5 | 12.2 | Sub-bass/bass |
| G5 (40-47) | 23.0 | 11.5 | 12.5 | Broadband body |
| G6 (48-55) | 39.3 | 19.0 | 19.6 | Sub-bass weight |
| G7 (56-63) | 96.4 | 54.5 | 62.6 | Air/brilliance |

### Individual Channel Impact Ranking (top 20, by composite score when zeroed)

| Rank | Channel | Impact | RMS% | Centroid% | Onset% | MFCC dist | Key effect |
|------|---------|--------|------|-----------|--------|-----------|------------|
| 1 | 19 | 106.7 | +5.9 | +11.4 | -0.8 | 88.5 | Presence/definition |
| 2 | 29 | 99.0 | +4.7 | -26.1 | +0.7 | 67.5 | Spectral tilt (low-shift) |
| 3 | 56 | 90.7 | +4.7 | +18.9 | +4.4 | 62.7 | Air/shimmer |
| 4 | 13 | 73.7 | -4.2 | +16.5 | +3.5 | 49.6 | HF texture |
| 5 | 14 | 32.2 | -7.5 | -4.6 | -1.6 | 18.6 | Full spectrum body |
| 6 | 23 | 27.3 | -8.7 | -9.4 | +1.1 | 8.1 | Low-mid warmth |
| 7 | 2 | 22.6 | -5.0 | +0.5 | +1.1 | 16.0 | Bass energy |
| 8 | 49 | 12.4 | +1.0 | -2.6 | +0.9 | 7.8 | Sub-bass texture |
| 9 | 55 | 9.9 | -2.7 | -0.3 | -0.1 | 6.8 | Presence damper |
| 10 | 41 | 9.5 | +0.2 | -2.0 | -0.8 | 6.6 | Upper-mid damper |
| 11 | 5 | 9.2 | -0.3 | -0.6 | +0.4 | 7.9 | Upper-mid/brilliance |
| 12 | 27 | 8.7 | -3.1 | +0.2 | +0.3 | 5.1 | Broadband low |
| 13 | 25 | 7.6 | -3.1 | -0.9 | +0.0 | 3.5 | Bass/low-mid |
| 14 | 51 | 7.4 | -3.0 | -0.1 | +1.2 | 3.2 | Sub-bass/bass |
| 15 | 50 | 7.4 | -4.8 | +0.4 | +0.2 | 2.0 | Sub-bass energy |
| 16 | 31 | 7.2 | -2.6 | +0.3 | +0.5 | 3.8 | Low-frequency |
| 17 | 5 | 9.2 | -0.3 | -0.6 | +0.4 | 7.9 | Upper-mid/brilliance split |
| 18 | 17 | 6.9 | -3.3 | +0.6 | +1.6 | 1.4 | Sub-bass |
| 19 | 47 | 6.6 | -3.4 | +0.4 | -0.0 | 2.8 | Sub-bass/bass |
| 20 | 16 | 6.3 | -4.0 | +0.7 | +0.1 | 1.5 | Sub-bass |

### Channels with Minimal Impact (impact < 2.5 when zeroed)

Channels 0, 1, 3, 4, 10, 38, 44, 57, 60, 61, 62, 63 all have composite impact scores below 2.5, meaning zeroing any single one of these produces less than ~2% change in any metric. These are either redundant with neighboring channels or encode fine detail that's distributed across many channels.

## Methodology Notes

- **Caching**: ComfyUI caches the KSampler output (deterministic seed), so only VAEDecode reruns for each variant. This ensures every variant starts from the identical latent — the differences are purely from the channel scaling in the VAE decoder.
- **Single seed**: Results may vary with different prompts/seeds. This analysis was done with one musical style (neo-soul). Channels may have different relative importance for different genres.
- **Post-sampling only**: This tests the VAE decoder's mapping. The diffusion model during sampling may use channels differently, and the guidance system may compensate for perturbations. A follow-up study with the guidance mechanism is recommended.
- **Linear perturbation**: Only tested 0x, 0.5x, 1.5x scaling. Nonlinear effects (e.g., 3x boost, negative values) were not explored.

## Phase 2: Guidance During Diffusion (Per-Group & Per-Channel)

### Overview

Phase 1 tested post-sampling perturbation (VAE decode semantics). Phase 2 tests **guidance-based perturbation during diffusion** — the model is run twice per step (once normal, once with specific channels scaled to 0.5x in the input), and generation is steered along the difference: `output + guidance_scale * (eq_output - output)`.

Unlike Phase 1's earlier incorrect attempt that used the 4x16 band EQ node, this experiment uses a custom `LatentChannelGuidance` node targeting **arbitrary channel ranges** — the correct 8 groups of 8 channels and the 6 keystone channels individually.

### Methodology

- Same baseline as Phase 1: seed=31, 30s neo-soul, 8 steps euler/simple, CFG 1.0
- Channel perturbation: target channels scaled to **0.5x** during guidance pass
- Guidance scales tested: **1.0** and **2.0**
- 29 total experiments: 1 baseline + 16 group + 12 keystone

### Key Finding: Guidance Generally Tracks Post-Sampling But With Amplification

The overall behavioral classification comparing guidance (gs=1.0, 0.5x scale) to Phase 1 post-sampling (zero):

| Group | Channels | Guidance/Post Ratio | Behavior |
|-------|----------|---------------------|----------|
| G0 | 0-7 | 1.01 | **SIMILAR** |
| G1 | 8-15 | 0.73 | **SIMILAR** (slightly compensated) |
| G2 | 16-23 | 1.38 | **SIMILAR** (slightly amplified) |
| G3 | 24-31 | 0.45 | **COMPENSATE** |
| G4 | 32-39 | 1.32 | **SIMILAR** |
| G5 | 40-47 | 1.24 | **SIMILAR** |
| G6 | 48-55 | 0.72 | **SIMILAR** |
| G7 | 56-63 | 1.10 | **SIMILAR** |

*Ratio = guidance effect magnitude / post-sampling effect magnitude on RMS+centroid. <0.5 = compensate, >1.5 = amplify.*

**Critical insight**: Unlike the previous (incorrect) Phase 2 that lumped 16 channels together and found dramatic amplification/compensation, targeting the correct 8-channel groups reveals that **guidance mostly produces similar-magnitude effects to post-sampling**. The exception is G3 (spectral balance), where the model actively compensates.

### Per-Group Guidance Results

#### G0 (ch 0-7): Low-Frequency Foundation — SIMILAR
- **Guidance gs=1.0**: RMS -8.5%, centroid +2.1%, MFCC 11.5
- **Guidance gs=2.0**: RMS -15.2%, centroid +2.2%, MFCC 17.2
- **Phase 1 zero**: RMS -10.3%, centroid +0.2%, MFCC 19.3
- **Phase 1 half**: RMS -5.2%, centroid +0.3%, MFCC 9.5
- Bands (gs=1): sub-bass -17%, bass -14%, low-mid -32%, brilliance -23%
- **Analysis**: Guidance at gs=1 with 0.5x scale falls between Phase 1's half and zero effects. The model does not compensate — it faithfully steers away from low-frequency content. Notably, guidance produces a **broader** spectral reduction (brilliance -23%) vs post-sampling which was more focused on bass.

#### G1 (ch 8-15): Broadband Energy — SLIGHTLY COMPENSATED
- **Guidance gs=1.0**: RMS -2.3%, centroid +20.7%, MFCC 47.3, ZCR +26%
- **Guidance gs=2.0**: RMS -2.5%, centroid +36.4%, MFCC 74.2, brilliance +89%
- **Phase 1 zero**: RMS -18.1%, centroid +13.3%, MFCC 62.2
- **Phase 1 half**: RMS -7.8%, centroid +5.9%, MFCC 38.0
- **Analysis**: The model **strongly compensates on RMS** (only -2.3% vs Phase 1's -7.8% at half) but **amplifies spectral redistribution** (centroid +21% vs +5.9%). Guidance pushes energy from sub-bass (-39%) into brilliance (+30%), creating a much brighter timbral character than post-sampling. This group's guidance effect is primarily a **spectral tilt** rather than volume reduction.

#### G2 (ch 16-23): Mid/Presence Texture — SLIGHTLY AMPLIFIED
- **Guidance gs=1.0**: RMS -7.2%, centroid +10.1%, MFCC 46.7
- **Guidance gs=2.0**: RMS -3.9%, centroid +13.9%, MFCC 72.9, sub-bass -72%
- **Phase 1 zero**: RMS -6.8%, centroid +5.7%, MFCC 109.6
- **Phase 1 half**: RMS -4.5%, centroid +1.5%, MFCC 58.6
- Bands (gs=1): sub-bass -66%, bass +13%, mid -28%, upper-mid -15%, brilliance +8%
- **Analysis**: Guidance produces **slightly larger RMS effects** than post-sampling half, but **smaller MFCC distance** (46.7 vs 58.6). The massive sub-bass drop (-66% at gs=1!) is a distinctive guidance effect not seen in post-sampling. The model redistributes energy from sub-bass to bass/low-mid while reducing mid-presence — a more complex reshaping than simple scaling.

#### G3 (ch 24-31): Spectral Balance — COMPENSATES
- **Guidance gs=1.0**: RMS +5.6%, centroid -8.4%, MFCC 48.4
- **Guidance gs=2.0**: RMS +5.7%, centroid -10.0%, MFCC 64.1, upper-mid +107%
- **Phase 1 zero**: RMS +4.1%, centroid -27.2%, MFCC 82.4
- **Phase 1 half**: RMS +2.4%, centroid -17.0%, MFCC 44.0
- Bands (gs=1): bass +13%, low-mid +24%, mid +33%, upper-mid +59%, presence +31%
- **Analysis**: The model **compensates on centroid** (-8.4% guidance vs -17% post-sampling half), confirming this group's role as a spectral balance controller that the model actively tries to maintain. However, guidance produces **massive upper-mid boost** (+59% at gs=1, +107% at gs=2) that's qualitatively different from post-sampling. The inverse behavior (attenuating these channels boosts mids) is preserved in guidance but the model restrains the overall spectral tilt.

#### G4 (ch 32-39): Sub-bass/Bass Foundation — SIMILAR
- **Guidance gs=1.0**: RMS -8.8%, centroid +6.9%, MFCC 6.3
- **Guidance gs=2.0**: RMS -9.6%, centroid +13.1%, MFCC 12.6, upper-mid +47%
- **Phase 1 zero**: RMS -10.6%, centroid +1.2%, MFCC 7.3
- **Phase 1 half**: RMS -5.6%, centroid +0.9%, MFCC 3.4
- Bands (gs=1): sub-bass -24%, bass -15%, upper-mid +27%
- **Analysis**: Very similar to post-sampling behavior for RMS/MFCC. However, guidance introduces a **spectral tilt** effect (centroid +6.9% vs +0.9% post-sampling) by boosting upper-mids while cutting lows. The model doesn't just reduce bass — it compensatorily pushes energy upward.

#### G5 (ch 40-47): Broadband Body — SIMILAR
- **Guidance gs=1.0**: RMS -10.8%, centroid -2.5%, MFCC 9.8
- **Guidance gs=2.0**: RMS -14.7%, centroid -2.6%, MFCC 13.0, sub-bass -55%
- **Phase 1 zero**: RMS -8.3%, centroid -2.4%, MFCC 12.2
- **Phase 1 half**: RMS -4.3%, centroid -0.8%, MFCC 5.9
- Bands (gs=1): sub-bass -36%, bass -16%, mid -13%, ZCR -12%
- **Analysis**: Guidance **slightly amplifies** the RMS effect (-10.8% vs -4.3% half, -8.3% zero). Preserves the darkening character (centroid -2.5%). The ZCR **decrease** (-12%) is notable — guidance makes the output smoother/less percussive, unlike G1 guidance which increases ZCR.

#### G6 (ch 48-55): Sub-bass Dominance — SIMILAR (slightly compensated)
- **Guidance gs=1.0**: RMS -7.5%, centroid +6.1%, MFCC 9.7
- **Guidance gs=2.0**: RMS -13.4%, centroid +12.0%, MFCC 15.4, ZCR +29%
- **Phase 1 zero**: RMS -16.7%, centroid -2.0%, MFCC 16.5
- **Phase 1 half**: RMS -8.9%, centroid -0.2%, MFCC 8.0
- Bands (gs=1): sub-bass -47%, upper-mid +14%
- **Analysis**: RMS is similar to post-sampling, but guidance produces a **centroid increase** (+6.1%) where post-sampling had centroid decrease (-0.2%). The model compensatorily pushes spectral energy upward when guided away from sub-bass channels, creating a brighter-sounding result than simple sub-bass removal.

#### G7 (ch 56-63): Air & Brilliance — SIMILAR
- **Guidance gs=1.0**: RMS +5.1%, centroid +20.0%, MFCC 61.9, brilliance +76%, onset +11%
- **Guidance gs=2.0**: RMS +4.5%, centroid +34.6%, MFCC 87.3, brilliance +113%, ZCR +46%
- **Phase 1 zero**: RMS +4.2%, centroid +18.5%, MFCC 69.1
- **Phase 1 half**: RMS +5.6%, centroid +8.2%, MFCC 37.9
- Bands (gs=1): bass +25%, low-mid +25%, mid +16%, brilliance +76%
- **Analysis**: Nearly identical RMS behavior. Guidance faithfully reproduces this group's inverse behavior (attenuating channels → more brilliance). The **onset strength increase** (+11%) is a guidance-specific effect — the output becomes more rhythmically active, suggesting these channels also encode temporal dynamics. At gs=2, ZCR increases +46%, making the output significantly noisier/more textured.

### Keystone Channel Guidance Results

#### Channel 19 (Presence/Definition) — Guidance Partially Compensates
- **Guidance gs=1.0**: RMS +5.9%, centroid +6.9%, MFCC 43.3
- **Guidance gs=2.0**: RMS +5.9%, centroid +11.1%, MFCC 64.9
- **Phase 1 zero**: RMS +5.9%, centroid +11.4%, MFCC 88.5
- Bands (gs=1): bass +34%, low-mid +12%, upper-mid -22%, presence -16%
- **Analysis**: RMS is identical, but **MFCC is about half** of post-sampling (43 vs 89), indicating the model partially compensates on timbre. The guidance effect produces a distinctive bass-boost (+34%) not seen in post-sampling, suggesting the model redistributes energy to maintain overall balance when ch19 is perturbed. Doubling guidance scale barely changes RMS (still +5.9%) but deepens the spectral reshaping.

#### Channel 29 (Spectral Tilt) — Guidance Partially Compensates
- **Guidance gs=1.0**: RMS +5.2%, centroid -12.1%, MFCC 41.8, onset +6.3%
- **Guidance gs=2.0**: RMS +4.3%, centroid -15.2%, MFCC 59.4, onset +10.9%
- **Phase 1 zero**: RMS +4.7%, centroid -26.1%, MFCC 67.5
- Bands (gs=1): bass +15%, low-mid +21%, mid +21%, upper-mid +16%, presence +23%
- **Analysis**: The model **compensates on centroid** (-12% guidance vs -26% post-sampling), confirming ch29's role as a spectral tilt control the model tries to maintain. However, guidance introduces a **broadband mid/presence boost** (+15-23% across bass through presence) that's more uniform than post-sampling's narrower effect. Onset strength also increases (+6-11%), adding rhythmic energy.

#### Channel 56 (Air/Shimmer) — Guidance Amplifies
- **Guidance gs=1.0**: RMS +5.4%, centroid +13.0%, MFCC 48.8, brilliance +58%
- **Guidance gs=2.0**: RMS +4.4%, centroid +24.1%, MFCC 72.5, brilliance +97%, low-mid +59%
- **Phase 1 zero**: RMS +4.7%, centroid +18.9%, MFCC 62.7
- Bands (gs=1): low-mid +38%, mid +36%, brilliance +58%
- **Analysis**: One of the few keystones where guidance **amplifies** the effect. At gs=2, centroid shifts +24% (vs +19% post-sampling), and brilliance nearly doubles. The massive low-mid boost (+38-59%) is a guidance-emergent effect — the model overcompensates by adding mid-range body when steered away from ch56. This creates a simultaneous brilliance + warmth effect not achievable via post-sampling.

#### Channel 13 (HF Texture) — Guidance Amplifies Brightness
- **Guidance gs=1.0**: RMS -0.7%, centroid +17.0%, MFCC 36.4, brilliance +44%, ZCR +21%
- **Guidance gs=2.0**: RMS +2.1%, centroid +28.7%, MFCC 56.8, brilliance +78%
- **Phase 1 zero**: RMS -4.2%, centroid +16.5%, MFCC 49.6
- **Analysis**: Guidance **compensates on RMS** (-0.7% vs -4.2%) but **amplifies the brightening effect** (centroid +17% at gs=1, +29% at gs=2 vs +17% post-sampling). The brilliance boost (+44-78%) is larger than any post-sampling effect. The model maintains volume while dramatically shifting spectral content upward.

#### Channel 14 (Full Spectrum Body) — Guidance Compensates
- **Guidance gs=1.0**: RMS +0.0%, centroid +1.3%, MFCC 18.0, onset -4.1%
- **Guidance gs=2.0**: RMS +0.8%, centroid +0.6%, MFCC 25.3, sub-bass -44%
- **Phase 1 zero**: RMS -7.5%, centroid -4.6%, MFCC 18.6
- Bands (gs=1): sub-bass -24%, bass +14%, presence -15%, brilliance -16%
- **Analysis**: **Strong RMS compensation** (0.0% vs -7.5% post-sampling). The model completely maintains volume but creates an **internal spectral redistribution**: sub-bass moves to bass, while presence and brilliance drop. Chroma distance is high (0.10-0.12), indicating harmonic content changes. Onset strength decreases (-4 to -8%), making the output less rhythmically active.

#### Channel 23 (Low-Mid Warmth) — Guidance Compensates Slightly
- **Guidance gs=1.0**: RMS -6.6%, centroid -2.0%, MFCC 4.0
- **Guidance gs=2.0**: RMS -7.4%, centroid -2.0%, MFCC 3.9
- **Phase 1 zero**: RMS -8.7%, centroid -9.4%, MFCC 8.1
- Bands (gs=1): sub-bass -24%, mid -16%, presence +5%, brilliance +12%
- **Analysis**: Guidance produces **similar but smaller effects** than post-sampling (centroid -2.0% vs -9.4%). Notably, **doubling guidance scale barely changes the effect** (gs=2 MFCC 3.9 ≈ gs=1 MFCC 4.0), suggesting ch23's guidance effect saturates quickly. The warmth/brightness balance shift is preserved but muted.

### Guidance Scale Linearity

| Behavior | Groups/Channels |
|----------|----------------|
| **Roughly linear** (gs=2 ≈ 2× gs=1) | G0, G4, G5, G6, ch13 |
| **Sub-linear** (diminishing returns) | G3, G7, ch19, ch23, ch29 |
| **Super-linear** (gs=2 > 2× gs=1) | G1 (brilliance: +30% → +89%), G2 (sub-bass: -66% → -72%, but bass +13% → +26%) |

Most effects are roughly linear between gs=1 and gs=2. The inverse-behavior groups (G3, G7) and keystone channels involved in spectral balance (ch29, ch23) show diminishing returns, suggesting the model resists large spectral shifts.

### Comparison: Phase 2 Guidance vs Phase 1 Post-Sampling

| Effect | Post-Sampling (Phase 1) | Guidance (Phase 2) |
|--------|------------------------|-------------------|
| **Magnitude** | Proportional to scale factor | Similar magnitude at gs=1 with 0.5x scale |
| **RMS preservation** | Changes proportional to perturbation | Model often compensates (ch14: 0%, ch13: -0.7% vs -4.2%) |
| **Spectral character** | Direct removal/addition of frequency content | Redistribution — removing lows often adds highs and vice versa |
| **Emergent effects** | None (purely VAE decoder) | Onset/rhythm changes, cross-band energy redistribution |
| **Inverse groups (G3, G7)** | Strong inverse effects | Inverse behavior preserved but compensated (G3) or faithfully reproduced (G7) |
| **Saturation** | Linear | Some channels saturate quickly (ch23, ch14) |

### Key Differences From Previous (Incorrect) Phase 2

The previous Phase 2 used the 4x16 band EQ node, which lumped opposing groups together:
- **Old "Band 2" (ch 16-31)** combined G2 (presence removal) and G3 (inverse mid boost) — the mixed signals created apparent "timbral transformation" effects that were actually interference between opposing groups
- **Old "Band 4" (ch 48-63)** combined G6 (sub-bass) and G7 (brilliance) — the "brilliance amplifier" effect was primarily from G7, with G6's sub-bass removal creating misleading RMS patterns

With proper per-group targeting, the picture is clearer:
1. **Most groups behave similarly to post-sampling** (ratio 0.7-1.4)
2. **G3 is the only true compensator** (ratio 0.45) — the model actively resists spectral balance changes
3. **No group shows the extreme amplification** (3.8x) reported in the previous Phase 2 — that was an artifact of mixing opposing groups
4. **The real creative power is in spectral redistribution**, not magnitude amplification

### Implications for EQ Node Design

1. **The 8-group structure is validated for guidance**: Each group has a distinct, predictable guidance response. The proposed 6-band restructuring from Phase 1 remains the correct approach.

2. **G3 (ch 24-31) needs special handling**: The model compensates during guidance, so higher guidance_scale values are needed to achieve the same effect as post-sampling. Consider a 2x internal multiplier for this band.

3. **Keystone channels work well as individual guidance targets**: Even single-channel guidance produces measurable, musically meaningful effects (ch29 at gs=1: -12% centroid, ch56 at gs=1: +58% brilliance).

4. **Guidance produces spectral redistribution rather than simple scaling**: When the model is guided away from low-frequency channels, it compensatorily boosts upper frequencies (and vice versa). This makes guidance a more "musical" tool than post-sampling — it reshapes balance rather than just cutting content.

5. **Some keystone channels saturate quickly**: ch23 and ch14 show minimal difference between gs=1 and gs=2, so the UI should cap their effective guidance range lower.

6. **Onset/rhythm effects are guidance-specific**: Post-sampling doesn't change onset strength, but guidance can increase rhythmic activity (G7: +11%) or decrease it (ch14: -4%). This opens a new creative dimension.

## Files

- Phase 1 raw results: `output/channel_experiments/analysis/raw_results.json`
- Phase 1 deltas: `output/channel_experiments/analysis/deltas.json`
- Phase 1 summary: `output/channel_experiments/analysis/summary.txt`
- Phase 2 guidance results: `output/channel_guidance_experiments/analysis/raw_results.json`
- Phase 2 guidance deltas: `output/channel_guidance_experiments/analysis/deltas.json`
- Phase 2 guidance report: `output/channel_guidance_experiments/analysis/phase2_report.txt`
- Phase 1 audio files: `output/channel_exp/` (89 FLAC files)
- Phase 2 audio files: `output/channel_guidance/` (29 FLAC files)
- Experiment scripts: `latent_channel_experiment.py`, `latent_channel_guidance_experiment.py`
- Temp nodes (can be deleted): `custom_nodes/temp_channel_experiment/`
- Phase 3 raw results: `output/channel_phase3_experiments/analysis/raw_results.json`
- Phase 3 deltas: `output/channel_phase3_experiments/analysis/deltas.json`
- Phase 3 report: `output/channel_phase3_experiments/analysis/phase3_report.txt`
- Phase 3 audio files: `output/channel_phase3/` (297 FLAC files)
- Phase 3 experiment script: `latent_channel_phase3_experiment.py`

---

## Phase 3: Higher-Level Musical Dimensions

### Overview

Phase 3 extends the investigation beyond spectral/timbral EQ to test whether guidance-based steering can reliably control **musical-level** properties: rhythmic density, harmonic complexity, instrument layering, and arrangement dynamics. This phase uses the `ACEStep15GenerationSteering` and `ACEStep15KeystoneConfig` nodes with `guidance_scale=1.0`.

### Methodology

- **3 seeds** (31, 42, 73) × **3 genres** (neo-soul, EDM, acoustic folk) = 9 baselines
- **12 individual band experiments** (6 bands × {0.5, 1.5})
- **12 individual keystone experiments** (6 keystones × {0.5, 1.5})
- **8 combination experiments** (the most important — testing emergent effects)
- **297 total experiments** (including baselines)
- All experiments use the live steering nodes with internal sensitivity normalization

### New Phase 3 Metrics

In addition to all Phase 1/2 metrics, Phase 3 adds:

1. **Onset density** (onsets/second) — distinguishes "sharper attacks" from "more notes"
2. **Onset regularity** (std of inter-onset intervals) — low = metronomic, high = syncopated
3. **Chroma flux** (harmonic change rate) — high = more chord movement
4. **Pitch class histogram entropy** — high = chromatic/complex, low = tonal/simple
5. **Spectral flatness** — high = noisy, low = tonal
6. **RMS variance** — high = dynamic, low = compressed
7. **Band onset correlation** — high = full-band hits, low = independent instrument layers

### Consistency Framework

Each finding is evaluated for:
- **Cross-seed consistency**: Same sign of change across ≥2/3 seeds?
- **Cross-genre consistency**: Same sign across ≥2/3 genres (neo-soul, EDM, folk)?
- Only effects consistent on BOTH dimensions (marked `***`) are considered reliable.

### Key Findings

#### Q1: Rhythm Control

The top rhythm-affecting controls (by onset density + regularity impact):

| Control | Impact | Reliable? | Effect |
|---------|--------|-----------|--------|
| ks_definition_hi (ch19↑) | 25.3 | Yes | Fewer onsets (-12%), more irregular (+13%) — consolidates rhythm |
| ks_definition_lo (ch19↓) | 20.9 | Partial | More onsets (+9%), opens up rhythm |
| ks_body_lo (ch14↓) | 16.3 | Yes (density) | +8% onset density, -9% band correlation → more independent layers |
| ks_body_hi (ch14↑) | 16.8 | Yes (density) | -9% onset density, +11% band correlation → consolidated, unified hits |
| combo_all_ks_07 | 23.9 | Yes | +14% onset density — most reliable rhythm densifier |
| combo_all_ks_13 | 28.8 | Yes | -12% onset density, +16% irregularity — rhythm simplifier |

**Key insight**: ch19 (definition) and ch14 (body) are the primary rhythm controls. Reducing definition opens up rhythmic density; boosting it consolidates. Body (ch14) controls whether instruments hit together or independently.

#### Q2: Harmonic Complexity Control

| Control | Impact | Reliable? | Effect |
|---------|--------|-----------|--------|
| band_weight_hi (ch48-55↑) | 10.0 | Yes | -8% chroma flux, -2% pitch entropy — harmonically simpler |
| ks_definition_hi (ch19↑) | 7.1 | Yes | -6% chroma flux — fewer harmonic changes |
| band_texture_hi (ch16-23↑) | 6.8 | Yes | -5% chroma flux, -1.5% entropy — reduces harmonic complexity |
| band_air_hi (ch56-63↑) | 4.4 | Yes | -4% chroma flux — subtle harmonic simplification |
| combo_weight15_foundation05 | 9.2+1.7 | Yes | -9% chroma flux — strongest harmonic simplifier |
| combo_foundation05_weight05 | 5.9+0.7 | Yes | +6% chroma flux, +0.7% entropy — harmonic enricher |

**Key insight**: Boosting weight/texture/definition reduces harmonic complexity (fewer chord changes, more tonal). Reducing foundation+weight enriches harmonics. The weight↑+foundation↓ combo is the strongest harmonic simplifier.

#### Q3: Emergent Combination Effects

Interaction scores (observed effect − sum of individual effects) reveal genuinely emergent behavior:

| Combination | Metric | Interaction | Interpretation |
|-------------|--------|-------------|----------------|
| texture↑+balance↓ | onset_regularity | -26.5 | Much more regular rhythm than either alone — emergent groove lock |
| weight↑+foundation↓ | spectral_flatness | -22.5 | Disproportionately more tonal than sum of parts |
| body↓+air↑ | spectral_flatness | -22.4 | Emergent tonal focus |
| air↓+body↓ | spectral_flatness | +20.1 | Emergent noisiness/texture |
| body↓+air↑ | rms_variance | +10.2 | More dynamic than expected — emergent breathing |
| texture↑+balance↓ | rms_variance | +10.0 | More dynamic range |

**Key insight**: Combinations produce genuinely non-linear effects. The texture↑+balance↓ combo creates an emergent "groove lock" (much more regular rhythm) that neither channel produces alone. Body↓+air↑ creates emergent dynamics.

#### Q4: Reliable Controls (Generalize Across Seeds AND Genres)

The most reliably controllable musical dimensions, ranked by effect size:

**Spectral flatness** (tonal vs. noisy) is the most reliably controllable property:
- ks_spectral_tilt_hi: +469% flatness (→ noisy/breathy)
- band_balance_hi: +224% flatness
- band_weight_hi / band_foundation_hi: -54%/-51% flatness (→ more tonal)

**RMS variance** (dynamics) is the second most controllable:
- ks_spectral_tilt_hi: -57% (→ compressed/flat dynamics)
- ks_body_hi: +37% (→ more dynamic)
- combo_foundation05_weight05: -32% (→ compressed)

**Onset density** (rhythmic activity):
- combo_all_ks_07: +14% (more events — reliable across all seeds and genres)
- ks_definition_hi: -12% (fewer events)
- combo_all_ks_13: -12% (fewer events)

**Band onset correlation** (instrument independence):
- band_balance_hi: -17% (→ more independent instrument layers)
- ks_body_hi: +11% (→ more unified/correlated hits)
- band_balance_lo: +9% (→ more unified)

#### Q5: Instrument Layering

Band onset correlation measures whether frequency bands trigger independently (low = layered arrangement with independent instruments) or together (high = full-band hits).

| Control | Correlation Change | Reliable? | Interpretation |
|---------|-------------------|-----------|----------------|
| band_balance_hi (ch24-31↑) | -17% | Yes | More independent layers |
| ks_body_hi (ch14↑) | +11% | Yes | More unified/correlated |
| ks_definition_hi (ch19↑) | +10% | Yes | More unified |
| band_balance_lo (ch24-31↓) | +9% | Yes | More unified |
| ks_body_lo (ch14↓) | -9% | Yes | More independent layers |
| band_weight_hi (ch48-55↑) | -8% | Yes | More independent layers |
| band_foundation_hi (ch0-7,32-39↑) | -7% | Yes | More independent layers |

**Key insight**: Balance band (ch24-31) is the primary layer independence control. Boosting it decouples instruments; reducing unifies them. Body keystone (ch14) has the inverse effect. This is a genuinely musical-level property — not just EQ.

### Summary: Musical Control Palette

| Musical Property | Primary Control | Direction | Reliable? |
|-----------------|----------------|-----------|-----------|
| More rhythmic events | combo_all_ks_07 (all keystones at 0.7) | ↑ density | Yes |
| Fewer rhythmic events | ks_definition_hi or combo_all_ks_13 | ↓ density | Yes |
| More regular rhythm | combo_texture15_balance05 | ↓ irregularity | Emergent |
| Simpler harmonics | combo_weight15_foundation05 | ↓ chroma flux | Yes |
| Richer harmonics | combo_foundation05_weight05 | ↑ chroma flux | Yes |
| More tonal | band_weight_hi or band_foundation_hi | ↓ flatness | Yes |
| More noisy/breathy | ks_spectral_tilt_hi | ↑ flatness | Yes |
| More dynamic | ks_body_hi | ↑ RMS variance | Yes |
| More compressed | ks_spectral_tilt_hi or combo_foundation05_weight05 | ↓ RMS variance | Yes |
| More instrument layers | band_balance_hi or ks_body_lo | ↓ band correlation | Yes |
| More unified hits | ks_body_hi or band_balance_lo | ↑ band correlation | Yes |
