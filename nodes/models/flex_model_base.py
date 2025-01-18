from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ...tooltips import apply_tooltips
from ..flex.flex_base import FlexBase
from comfy.ldm.modules.attention import optimized_attention, BasicTransformerBlock, default
import torch
import math

#NOTE: TOTALLY EXPERIMENTAL

class FlexFeatureAttentionControl:
    @classmethod

    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "feature": ("FEATURE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "feature_mode": (["relative", "absolute"],),
            "attention_type": (["cross", "self", "both"],),
            "scaling_mode": (["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],),
            "layer_selection": (["all", "first_half", "second_half", "custom"],),
            "custom_layers": ("STRING", {"default": "1,2,3", "multiline": False}),
            "auto_projection": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "RyanOnTheInside/FlexFeatures/ModelModulation"

    def patch(self, model, feature, strength, feature_mode, attention_type, scaling_mode, layer_selection, custom_layers, auto_projection):
        print("\n=== FlexFeatureAttentionControl Setup ===")
        print(f"Attention type: {attention_type}")
        print(f"Scaling mode: {scaling_mode}")
        print(f"Feature mode: {feature_mode}")
        print(f"Strength: {strength}")
        print(f"Layer selection: {layer_selection}")
        print(f"Feature frame count: {feature.frame_count}")
        print(f"Auto projection: {auto_projection}")
        print(f"Feature values: {[feature.get_value_at_frame(i) for i in range(feature.frame_count)]}")

        m = model.clone()

        # Create projection layers for dimension mismatch
        projections = {}

        def get_target_layers(model):
            """Helper to get target attention layers based on selection"""
            layers = []
            
            def register_attn_block(name, module):
                if isinstance(module, BasicTransformerBlock):
                    if attention_type == "cross" and module.attn2 is not None:
                        layers.append((name, module.attn2))
                    elif attention_type == "self" and not module.disable_self_attn:
                        layers.append((name, module.attn1))
                    elif attention_type == "both":
                        if not module.disable_self_attn:
                            layers.append((name + ".attn1", module.attn1))
                        if module.attn2 is not None:
                            layers.append((name + ".attn2", module.attn2))

            # Traverse model hierarchy
            for name, module in model.model.diffusion_model.named_modules():
                register_attn_block(name, module)
            
            print(f"\nFound {len(layers)} matching attention layers")
            for layer_name, _ in layers:
                print(f"- {layer_name}")
            
            total_layers = len(layers)
            if layer_selection == "first_half":
                selected = layers[:total_layers//2]
            elif layer_selection == "second_half":
                selected = layers[total_layers//2:]
            elif layer_selection == "custom":
                try:
                    indices = [int(i.strip()) for i in custom_layers.split(",")]
                    selected = [layers[i] for i in indices if 0 <= i < total_layers]
                except:
                    print("Invalid custom layer indices, using all layers")
                    selected = layers
            else:
                selected = layers
                
            print(f"\nSelected {len(selected)} layers for patching")
            for layer_name, _ in selected:
                print(f"* {layer_name}")
                
            return selected

        def create_projection_layer(in_dim, out_dim, layer_name):
            """Create a projection layer for dimension mismatch"""
            if (in_dim, out_dim, layer_name) not in projections:
                # Get device and dtype from model parameters
                device = next(model.model.parameters()).device
                dtype = next(model.model.parameters()).dtype
                
                projections[(in_dim, out_dim, layer_name)] = torch.nn.Linear(in_dim, out_dim)
                projections[(in_dim, out_dim, layer_name)].to(device=device, dtype=dtype)
                print(f"Created projection layer for {layer_name}: {in_dim} -> {out_dim}")
            return projections[(in_dim, out_dim, layer_name)]

        def flex_attention(q, k, v, extra_options):
            """
            Apply flexible attention scaling based on the current mode and settings
            """
            # Store original tensors for comparison
            k_orig = k.clone()
            v_orig = v.clone()
            
            print("\n=== FlexFeatureAttentionControl Debug ===")
            print(f"Input shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
            
            # Handle dimension mismatch if auto_projection is enabled
            if auto_projection:
                expected_dim = q.shape[-1]
                if k.shape[-1] != expected_dim:
                    print(f"K dimension mismatch: {k.shape[-1]} vs {expected_dim}")
                    proj_k = create_projection_layer(k.shape[-1], expected_dim, f"k_proj_{k.shape[-1]}_{expected_dim}")
                    k = proj_k(k)
                    print(f"K shape after projection: {k.shape}")
                
                if v.shape[-1] != expected_dim:
                    print(f"V dimension mismatch: {v.shape[-1]} vs {expected_dim}")
                    proj_v = create_projection_layer(v.shape[-1], expected_dim, f"v_proj_{v.shape[-1]}_{expected_dim}")
                    v = proj_v(v)
                    print(f"V shape after projection: {v.shape}")
            
            # Get current step from sigma if available
            sigmas = extra_options.get("sigmas", None)
            if sigmas is not None:
                sigma = sigmas.detach().cpu()[0].item()
                total_steps = len(sigmas)
                current_step = total_steps - len([s for s in sigmas if s >= sigma])
                print(f"Step {current_step}/{total_steps} (sigma: {sigma:.4f})")
                
                # Map step to feature frame with interpolation
                frame_float = (current_step / total_steps) * (feature.frame_count - 1)
            else:
                # If no sigmas available, use middle frame
                print("No step information available, using middle frame")
                frame_float = (feature.frame_count - 1) / 2
            
            frame_low = int(frame_float)
            frame_high = min(frame_low + 1, feature.frame_count - 1)
            alpha = frame_float - frame_low
            
            # Interpolate feature values
            value_low = feature.get_value_at_frame(frame_low)
            value_high = feature.get_value_at_frame(frame_high)
            feature_value = value_low * (1 - alpha) + value_high * alpha
            print(f"Frame {frame_float:.2f} value: {feature_value:.4f} (interpolated between {value_low:.4f} and {value_high:.4f})")

            # Calculate scale based on feature mode with improved stability
            if feature_mode == "relative":
                # For relative mode, limit the maximum relative change
                clamped_feature = max(-0.75, min(0.75, feature_value))  # Increased range from [-0.5, 0.5] to [-0.75, 0.75]
                scale = 1.0 + (clamped_feature * strength * 2.0)  # Doubled strength impact
                print(f"Relative scale: 1.0 + (clamp({feature_value:.4f}) * {strength:.4f} * 2.0) = {scale:.4f}")
            else:
                # For absolute mode, ensure we don't scale too far from 1.0
                clamped_feature = max(0.0, min(1.0, feature_value))  # Clamp to [0, 1]
                base_scale = 0.5 + (clamped_feature * strength)  # Center around 0.5 for more dynamic range
                scale = max(0.1, min(2.0, base_scale))  # Wider scale range
                print(f"Absolute scale: clamp(0.5 + ({feature_value:.4f} * {strength:.4f})) = {scale:.4f}")

            # Get attention precision from layer if available
            attn_precision = None
            if hasattr(layer, 'attn_precision'):
                attn_precision = layer.attn_precision
                print(f"Using attn_precision: {attn_precision}")

            # Apply scaling based on selected mode with improved stability
            print(f"Applying {scaling_mode} scaling:")
            if scaling_mode == "K+mean(V) w/ C penalty":
                # Use sqrt for gentler context penalty
                context_size = float(k.shape[2])
                scaling = math.sqrt(context_size / 1280.0)  # Base size of 1280 for reference
                scale = scale * (scaling * 0.75 + 0.25)  # Blend with identity to prevent extreme scaling
                print(f"Scale after context penalty: {scale:.4f}")
                k = k * scale
                v_mean = torch.mean(v, dim=1, keepdim=True)
                v = v + (v_mean * (scale - 1.0))  # Removed the 0.5 reduction factor
            elif scaling_mode == "K+V":
                k = k * scale
                v = v * scale
            elif scaling_mode == "K+V w/ C penalty":
                # Use sqrt for gentler context penalty
                context_size = float(k.shape[2])
                scaling = math.sqrt(context_size / 1280.0)  # Base size of 1280 for reference
                scale = scale * (scaling * 0.75 + 0.25)  # Blend with identity to prevent extreme scaling
                print(f"Scale after context penalty: {scale:.4f}")
                k = k * scale
                v = v * scale
            else:  # "V only"
                v = v * scale
            
            # Log the effect of modifications
            print(f"K change: mean diff = {(k - k_orig).abs().mean().item():.4f}")
            print(f"V change: mean diff = {(v - v_orig).abs().mean().item():.4f}")

            # Return attention result
            return optimized_attention(q, k, v, extra_options["n_heads"])

        # Get target layers to patch
        target_layers = get_target_layers(m)
        
        # Apply patches to target layers
        for layer_name, layer in target_layers:
            original_forward = layer.forward
            def make_patch(orig_forward):
                def patched_forward(x, context=None, value=None, mask=None):
                    # Get transformer options from x if available
                    transformer_options = {}
                    if hasattr(x, 'transformer_options'):
                        transformer_options = x.transformer_options

                    print("\n=== Debug: Tensor Shapes ===")
                    print(f"Input x shape: {x.shape}")
                    if context is not None:
                        print(f"Input context shape: {context.shape}")
                    if value is not None:
                        print(f"Input value shape: {value.shape}")

                    # Store original input dimensions and tensor
                    original_dim = x.shape[-1]
                    x_orig = x
                    needs_projection = auto_projection and original_dim != layer.to_q.in_features

                    # Handle spatial tensor reshaping
                    if len(x.shape) == 4:
                        b, c, h, w = x.shape
                        print(f"Reshaping spatial tensor - Original: {x.shape}")
                        x = x.movedim(1, 3).flatten(1, 2)  # (b, h, w, c) -> (b, h*w, c)
                        print(f"After reshape: {x.shape}")
                        if context is not None and len(context.shape) == 4:
                            print(f"Reshaping context - Original: {context.shape}")
                            context = context.movedim(1, 3).flatten(1, 2)
                            print(f"After reshape: {context.shape}")
                        if value is not None and len(value.shape) == 4:
                            print(f"Reshaping value - Original: {value.shape}")
                            value = value.movedim(1, 3).flatten(1, 2)
                            print(f"After reshape: {value.shape}")

                    # Handle input dimension mismatch before q projection
                    if needs_projection:
                        print(f"\nInput dimension mismatch for Q projection: {x.shape[-1]} vs {layer.to_q.in_features}")
                        proj_x = create_projection_layer(x.shape[-1], layer.to_q.in_features, f"x_proj_{x.shape[-1]}_{layer.to_q.in_features}")
                        x_projected = proj_x(x)
                        print(f"X shape after projection: {x_projected.shape}")
                    else:
                        x_projected = x
                    
                    print("\nProjecting q...")
                    q = layer.to_q(x_projected)
                    print(f"q shape after projection: {q.shape}")
                    
                    context = default(context, x_projected)
                    # Handle context dimension mismatch before k projection
                    if auto_projection and context.shape[-1] != layer.to_k.in_features:
                        print(f"\nContext dimension mismatch for K projection: {context.shape[-1]} vs {layer.to_k.in_features}")
                        proj_context = create_projection_layer(context.shape[-1], layer.to_k.in_features, f"context_proj_{context.shape[-1]}_{layer.to_k.in_features}")
                        context = proj_context(context)
                        print(f"Context shape after projection: {context.shape}")
                    
                    print(f"\nContext shape before k projection: {context.shape}")
                    k = layer.to_k(context)
                    print(f"k shape after projection: {k.shape}")
                    
                    if value is not None:
                        # Handle value dimension mismatch before v projection
                        if auto_projection and value.shape[-1] != layer.to_v.in_features:
                            print(f"\nValue dimension mismatch for V projection: {value.shape[-1]} vs {layer.to_v.in_features}")
                            proj_value = create_projection_layer(value.shape[-1], layer.to_v.in_features, f"value_proj_{value.shape[-1]}_{layer.to_v.in_features}")
                            value = proj_value(value)
                            print(f"Value shape after projection: {value.shape}")
                        
                        print(f"\nValue shape before v projection: {value.shape}")
                        v = layer.to_v(value)
                    else:
                        print(f"\nUsing context for v projection: {context.shape}")
                        v = layer.to_v(context)
                    print(f"v shape after projection: {v.shape}")
                    
                    # Apply our custom attention
                    if mask is None:
                        extra_options = {
                            "n_heads": layer.heads,
                            "sigmas": transformer_options.get("sigmas", None)
                        }
                        print("\n=== Calling flex_attention ===")
                        print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
                        print(f"n_heads: {extra_options['n_heads']}")
                        out = flex_attention(q, k, v, extra_options)
                        print(f"flex_attention output shape: {out.shape}")
                    else:
                        print("\n=== Using original forward (masked) ===")
                        return orig_forward(x, context, value, mask)
                    
                    print("\n=== Output Processing ===")
                    print(f"Shape before output projection: {out.shape}")
                    out = layer.to_out(out)
                    print(f"Shape after output projection: {out.shape}")

                    # Project output back to original dimension if needed
                    if needs_projection:
                        print(f"\nProjecting output back to original dimension: {out.shape[-1]} -> {original_dim}")
                        proj_out = create_projection_layer(out.shape[-1], original_dim, f"out_proj_{out.shape[-1]}_{original_dim}")
                        out = proj_out(out)
                        print(f"Final output shape: {out.shape}")

                    # Reshape back to spatial if needed
                    if len(x.shape) == 4:
                        print(f"\nReshaping back to spatial - Before: {out.shape}")
                        out = out.view(b, h, w, -1).movedim(3, 1)  # (b, h*w, c) -> (b, c, h, w)
                        print(f"After spatial reshape: {out.shape}")
                    
                    print("=======================================")
                    # Apply residual connection in original space
                    return x_orig + out
                return patched_forward
            
            layer.forward = make_patch(original_forward)

        return (m,)

    @classmethod
    def IS_CHANGED(cls, model, feature, **kwargs):
        """Track model and feature changes"""
        return float(feature.get_hash() + hash(model))