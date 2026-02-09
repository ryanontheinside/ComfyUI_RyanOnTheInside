import folder_paths
import os
import av
import torch


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


class PreviewAudioCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO",),
                "audio_b": ("AUDIO",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "RyanOnTheInside/Audio"

    def compare(self, audio_a, audio_b):
        result_a = _save_audio_temp(audio_a, "comfy_ab_a")
        result_b = _save_audio_temp(audio_b, "comfy_ab_b")
        return {"ui": {"a_audio": [result_a], "b_audio": [result_b]}}

NODE_CLASS_MAPPINGS = {
    "PreviewAudioCompare": PreviewAudioCompare,
}
