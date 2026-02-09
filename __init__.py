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


import importlib
import logging

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
_NODE_MODULES = [
    ".nodes.misc.misc_nodes",
    ".nodes.masks.temporal_masks",
    ".nodes.audio.audio_nodes",
    ".nodes.audio.flex_audio_visualizer",
    ".nodes.audio.audio_nodes_effects",
    ".nodes.audio.audio_compare",
    ".nodes.audio.audio_nodes_utility",
    ".nodes.audio.midi_nodes",
    ".nodes.audio.flex_audio",
    ".nodes.flex.feature_extractors",
    ".nodes.flex.feature_extractors_whisper",
    ".nodes.flex.feature_extractors_audio",
    ".nodes.flex.feature_extractors_midi",
    ".nodes.flex.feature_extractors_proximity",
    ".nodes.flex.visualizers",
    ".nodes.flex.flex_externals",
    ".nodes.flex.feature_modulation",
    ".nodes.flex.feature_pipe",
    ".nodes.flex.parameter_scheduling",
    ".nodes.masks.optical_flow_masks",
    ".nodes.masks.particle_system_masks",
    ".nodes.masks.mask_utility_nodes",
    ".nodes.masks.flex_masks",
    ".nodes.images.flex_images",
    ".nodes.images.image_utility_nodes",
    ".nodes.video.flex_video",
    ".nodes.depth.depth_base",
    ".nodes.utility_nodes",
    ".nodes.latents.flex_latents",
    ".nodes.latents.latent_frequency_blender",
    ".nodes.doom.doom",
    ".audio_latent_blend",
    ".external_integration",
    ".nodes.acestep.nodes",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod_path in _NODE_MODULES:
    try:
        _mod = importlib.import_module(_mod_path, package=__name__)
        if hasattr(_mod, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
        if hasattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as _e:
        logging.warning(f"[RyanOnTheInside] Failed to load {_mod_path}: {_e}")

WEB_DIRECTORY = "./web/js"
EXTENSION_WEB_DIRS = ["./web/extensions"]

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
