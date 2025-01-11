from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ...tooltips import apply_tooltips
from ..flex.flex_base import FlexBase
from comfy.ldm.modules.attention import optimized_attention

@apply_tooltips
class FlexModelBase(FlexBase):
    """Base class for nodes that modify models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "RyanOnTheInside/FlexFeatures/ModelModulation"

    def apply_effect(self, model, strength, feature_threshold=0.0, 
                    feature_param="None", feature_mode="relative", 
                    opt_feature=None, **kwargs):
        """Common model handling logic"""
        
        # Get feature value if provided
        feature_value = None
        if opt_feature is not None:
            feature_value = opt_feature.get_value_at_frame(0)
            
            # Apply threshold
            if feature_value < feature_threshold:
                return (model,)

        # Apply the effect
        return self.apply_effect_internal(
            model=model,
            feature_value=feature_value,
            strength=strength,
            feature_param=feature_param,
            feature_mode=feature_mode,
            **kwargs
        )

    @abstractmethod
    def apply_effect_internal(self, model, feature_value, strength, feature_param, feature_mode, **kwargs):
        """To be implemented by subclasses"""
        pass

    def _get_model_layers(self, model, layer_type=None):
        """Helper to traverse model and get layers of specified type"""
        layers = []
        
        # Get the diffusion model
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diff_model = model.model.diffusion_model
            
            # Traverse all named modules
            for name, module in diff_model.named_modules():
                if layer_type is None:
                    layers.append((name, module))
                elif layer_type == "attention" and "attn" in name:
                    layers.append((name, module))
                elif layer_type == "cross_attention" and "attn2" in name:
                    layers.append((name, module))
                elif layer_type == "self_attention" and "attn1" in name:
                    layers.append((name, module))
                elif layer_type == "conv" and any(x in name for x in ["conv", "Conv"]):
                    layers.append((name, module))
                
        return layers

    def _get_layer_subset(self, layers, selection, custom_indices=None):
        """Helper to get subset of layers based on selection type"""
        total_layers = len(layers)
        
        if selection == "custom" and custom_indices:
            try:
                indices = [int(i.strip()) for i in custom_indices.split(",")]
                return [layers[i] for i in indices if 0 <= i < total_layers]
            except (ValueError, IndexError):
                print(f"Warning: Invalid custom layer indices. Using all layers.")
                return layers
        
        if selection == "first_half":
            return layers[:total_layers//2]
        elif selection == "second_half":
            return layers[total_layers//2:]
        else:  # "all" or default
            return layers

    def _store_original_param(self, module, param_name, store_name=None):
        """Helper to store original parameter values"""
        if store_name is None:
            store_name = f'original_{param_name}'
            
        if not hasattr(module, store_name):
            setattr(module, store_name, getattr(module, param_name, None))

    def _restore_original_param(self, module, param_name, store_name=None):
        """Helper to restore original parameter values"""
        if store_name is None:
            store_name = f'original_{param_name}'
            
        if hasattr(module, store_name):
            original_value = getattr(module, store_name)
            if original_value is not None:
                setattr(module, param_name, original_value)
            delattr(module, store_name)

@apply_tooltips
class FlexCrossAttentionModulator(FlexModelBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "layer_selection": (["all", "first_half", "second_half", "custom"],),
                "custom_layers": ("STRING", {"default": "1,2,3", "multiline": False}),
                "max_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "min_scale": ("FLOAT", {"default": 0.1, "min": 0.1, "max": 10.0, "step": 0.1}),
                "embeds_scaling": (["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],),
            }
        }
    
    FUNCTION = "apply_effect"

    @classmethod
    def get_modifiable_params(self):
        return ["attention_scale", "None"]

    def apply_effect_internal(self, model, feature_value, strength, layer_selection, 
                            custom_layers, max_scale, min_scale, feature_param, 
                            feature_mode, embeds_scaling='V only', **kwargs):

        # Get target layers
        target_layers = self._get_layer_subset(
            self._get_model_layers(model, "cross_attention"),
            layer_selection,
            custom_layers
        )

        # Create attention patch
        patch = Attn2Replace(self._create_attention_callback(
            feature_value=feature_value,
            strength=strength,
            max_scale=max_scale,
            min_scale=min_scale,
            embeds_scaling=embeds_scaling
        ))

        # Apply patch to model
        for name, layer in target_layers:
            self._store_original_param(layer, 'scale')
            if not hasattr(layer, 'attention_patch'):
                layer.attention_patch = patch
            else:
                layer.attention_patch.add(patch.callback[0], **patch.kwargs[0])

        return (model,)

    def _create_attention_callback(self, feature_value, strength, max_scale, min_scale, embeds_scaling):
        def callback(out, q, k, v, extra_options, **kwargs):
            # Calculate scale based on feature value
            scale = min_scale + (max_scale - min_scale) * feature_value
            scale = scale * strength

            if embeds_scaling == 'K+mean(V) w/ C penalty':
                scaling = float(k.shape[2]) / 1280.0
                scale = scale * scaling
                k = k * scale
                v_mean = torch.mean(v, dim=1, keepdim=True)
                v = (v - v_mean) + v_mean * scale
            elif embeds_scaling == 'K+V':
                k = k * scale
                v = v * scale
            elif embeds_scaling == 'K+V w/ C penalty':
                scaling = float(k.shape[2]) / 1280.0
                scale = scale * scaling
                k = k * scale
                v = v * scale
            else:  # 'V only'
                v = v * scale

            return out + optimized_attention(q, k, v, extra_options["n_heads"])

        return callback

class Attn2Replace:
    def __init__(self, callback=None, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **self.kwargs[i])

        return out.to(dtype=dtype)