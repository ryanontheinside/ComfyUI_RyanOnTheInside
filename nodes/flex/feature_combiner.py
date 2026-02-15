from ... import RyanOnTheInside
import numpy as np
import torch
import json
import folder_paths
import os
import av
from .feature_modulation import FeatureModulationBase
from ...tooltips import apply_tooltips
from scipy.interpolate import interp1d


def _save_audio_temp(audio, prefix):
    """Save audio dict to a temp flac file, return SavedResult-style dict."""
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_temp_directory()
    )
    file = f"{filename}_{counter:05}_.flac"
    output_path = os.path.join(full_output_folder, file)

    waveform = audio["waveform"].cpu().squeeze(0)  # [channels, samples]
    sample_rate = audio["sample_rate"]
    layout = "mono" if waveform.shape[0] == 1 else "stereo"

    container = av.open(output_path, "w")
    stream = container.add_stream("flac", rate=sample_rate, layout=layout)

    frame = av.AudioFrame.from_ndarray(
        waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
        format="flt",
        layout=layout,
    )
    frame.sample_rate = sample_rate
    frame.pts = 0

    container.mux(stream.encode(frame))
    container.mux(stream.encode(None))
    container.close()

    return {"filename": file, "subfolder": subfolder, "type": "temp"}


@apply_tooltips
class AdvancedFeatureCombiner(FeatureModulationBase):
    CATEGORY = "RyanOnTheInside/FlexFeatures/FeatureModulators"
    FUNCTION = "combine"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combine_mode": (["weighted_sum", "normalized", "max", "min", "multiply", "subtract", "divide"],),
                "interpolation_method": (["linear", "cubic", "nearest", "hold"],),
                "frame_count": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "weight_data": ("STRING", {"default": "{}"}),
                **super().INPUT_TYPES()["required"],
            },
            "optional": {
                "audio": ("AUDIO",),
                "feature_1": ("FEATURE", {"forceInput": True}),
                "feature_2": ("FEATURE", {"forceInput": True}),
                "feature_3": ("FEATURE", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)

    def _interpolate_weights(self, points, frame_count, method):
        """Interpolate weight control points into a per-frame weight array."""
        if not points or len(points) == 0:
            return np.ones(frame_count)

        points = sorted(points, key=lambda p: p[0])
        frames = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])

        if len(points) == 1:
            return np.full(frame_count, values[0])

        x_out = np.arange(frame_count)

        if method == "hold":
            result = np.zeros(frame_count)
            for i in range(len(frames) - 1):
                start = int(frames[i])
                end = int(frames[i + 1])
                result[start:end] = values[i]
            result[int(frames[-1]):] = values[-1]
            if int(frames[0]) > 0:
                result[:int(frames[0])] = values[0]
            return result

        kind = method
        if kind == "cubic" and len(frames) < 4:
            kind = "quadratic" if len(frames) >= 3 else "linear"

        fill = (values[0], values[-1])
        f = interp1d(frames, values, kind=kind, bounds_error=False, fill_value=fill)
        return np.clip(f(x_out), 0.0, 1.0)

    def _compute_waveform_peaks(self, audio, num_peaks=500):
        """Downsample audio waveform to peak amplitudes for UI display."""
        waveform = audio["waveform"]
        mono = waveform[0].mean(dim=0).abs()
        samples = mono.numpy()
        total = len(samples)
        if total <= num_peaks:
            return samples.tolist()

        chunk_size = total / num_peaks
        peaks = []
        for i in range(num_peaks):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            peaks.append(float(np.max(np.abs(samples[start:end]))))
        return peaks

    def _downsample_feature(self, values, num_points=500):
        """Downsample feature values for UI display."""
        if len(values) <= num_points:
            return values
        step = len(values) / num_points
        return [float(values[int(i * step)]) for i in range(num_points)]

    def combine(self, combine_mode, interpolation_method, frame_count, frame_rate, weight_data, invert_output,
                audio=None, feature_1=None, feature_2=None, feature_3=None):

        # Collect connected features
        all_features = [feature_1, feature_2, feature_3]
        features = [(i + 1, f) for i, f in enumerate(all_features) if f is not None]

        # Determine frame_count
        fc = frame_count
        if fc == 0:
            if audio is not None:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
                duration = waveform.shape[-1] / sample_rate
                fc = int(duration * frame_rate)
            elif features:
                fc = features[0][1].frame_count
            else:
                fc = 30

        if not features:
            dummy_values = [0.0] * fc
            from .features import BaseFeature
            dummy = BaseFeature.__new__(BaseFeature)
            dummy.__dict__.update({
                'name': 'empty',
                'frame_count': fc,
                'frame_rate': frame_rate,
                'data': dummy_values,
            })
            processed = self.create_processed_feature(dummy, dummy_values, "Combined", invert_output)
            ui_data = {
                "waveform_peaks": [[]],
                "audio_duration": [0.0],
                "frame_count": [fc],
                "feature_data": [{}],
                "combined_data": [[]],
            }
            return {"ui": ui_data, "result": (processed,)}

        # Parse weight_data
        try:
            wd = json.loads(weight_data)
        except (json.JSONDecodeError, TypeError):
            wd = {}

        # Extract values and compute weight envelopes
        feature_values = []
        weight_envelopes = []
        feature_data_for_ui = {}

        for idx, feat in features:
            vals = [feat.get_value_at_frame(i) for i in range(min(feat.frame_count, fc))]
            if len(vals) < fc:
                vals.extend([vals[-1] if vals else 0.0] * (fc - len(vals)))
            else:
                vals = vals[:fc]
            feature_values.append(np.array(vals, dtype=np.float64))

            # Store downsampled feature data for UI
            feature_data_for_ui[str(idx)] = self._downsample_feature(vals)

            key = str(idx)
            if key in wd and wd[key]:
                envelope = self._interpolate_weights(wd[key], fc, interpolation_method)
            else:
                envelope = np.ones(fc)
            weight_envelopes.append(envelope)

        feature_values = np.array(feature_values)
        weight_envelopes = np.array(weight_envelopes)

        if combine_mode == "weighted_sum":
            combined = np.sum(feature_values * weight_envelopes, axis=0)
        elif combine_mode == "normalized":
            weighted = feature_values * weight_envelopes
            weight_sum = np.sum(weight_envelopes, axis=0)
            weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
            combined = np.sum(weighted, axis=0) / weight_sum
        elif combine_mode == "max":
            weighted = feature_values * weight_envelopes
            combined = np.max(weighted, axis=0)
        elif combine_mode == "min":
            weighted = feature_values * weight_envelopes
            combined = np.min(weighted, axis=0)
        elif combine_mode == "multiply":
            weighted = feature_values * weight_envelopes
            combined = np.prod(weighted, axis=0)
        elif combine_mode == "subtract":
            # First feature minus all others (all weighted)
            weighted = feature_values * weight_envelopes
            combined = weighted[0].copy()
            for i in range(1, len(weighted)):
                combined -= weighted[i]
        elif combine_mode == "divide":
            # First feature divided by each subsequent (all weighted)
            weighted = feature_values * weight_envelopes
            combined = weighted[0].copy()
            for i in range(1, len(weighted)):
                combined = np.where(weighted[i] != 0, combined / weighted[i], 0.0)
        else:
            combined = np.sum(feature_values * weight_envelopes, axis=0)

        combined_list = combined.tolist()

        base_feature = features[0][1]
        processed = self.create_processed_feature(base_feature, combined_list, "Combined", invert_output)

        # Build UI data - every value must be a list for ComfyUI's ui aggregation
        ui_data = {
            "frame_count": [fc],
            "feature_data": [feature_data_for_ui],
            "combined_data": [self._downsample_feature(combined_list)],
        }

        if audio is not None:
            peaks = self._compute_waveform_peaks(audio)
            sample_rate = audio["sample_rate"]
            duration = float(audio["waveform"].shape[-1]) / sample_rate
            ui_data["waveform_peaks"] = [peaks]
            ui_data["audio_duration"] = [duration]
            # Save audio to temp file for playback
            audio_result = _save_audio_temp(audio, "comfy_fc_audio")
            ui_data["audio_file"] = [audio_result]
        else:
            ui_data["waveform_peaks"] = [[]]
            ui_data["audio_duration"] = [0.0]
            ui_data["audio_file"] = [{}]

        return {"ui": ui_data, "result": (processed,)}


NODE_CLASS_MAPPINGS = {
    "AdvancedFeatureCombiner": AdvancedFeatureCombiner,
}
