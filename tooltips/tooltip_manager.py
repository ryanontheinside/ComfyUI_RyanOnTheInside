"""
Core tooltip management functionality.
"""
import logging
from typing import Dict, Optional, Union, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('tooltips')

class TooltipManager:
    """
    Manages tooltips for all nodes in the suite.
    Provides a centralized way to define and access tooltips.
    """
    
    # Dictionary mapping node names to their parameter tooltips
    NODE_TOOLTIPS = {}
    
    # Dictionary mapping node names to their parent classes
    INHERITANCE_MAP = {}
    
    @classmethod
    def get_tooltips(cls, node_class: str) -> dict:
        """
        Get all tooltips for a node class, including inherited tooltips.
        
        Args:
            node_class: Name of the node class
            
        Returns:
            Dictionary mapping parameter names to tooltip descriptions
        """
        tooltips = {}
        
        # Build inheritance chain
        inheritance_chain = []
        current = node_class
        while current in cls.INHERITANCE_MAP:
            parent = cls.INHERITANCE_MAP[current]
            if isinstance(parent, list):
                inheritance_chain.extend(parent)
                current = parent[0]  # For simplicity, follow first parent's chain
            else:
                inheritance_chain.append(parent)
                current = parent
        
        logger.info(f"Inheritance chain for {node_class}: {inheritance_chain}")
        
        # Apply tooltips in reverse order (base class first)
        for parent in reversed(inheritance_chain):
            if parent in cls.NODE_TOOLTIPS:
                logger.info(f"Adding tooltips from parent {parent}: {cls.NODE_TOOLTIPS[parent]}")
                tooltips.update(cls.NODE_TOOLTIPS[parent])
            else:
                logger.info(f"Parent {parent} has no tooltips registered - skipping")
        
        # Apply class's own tooltips last (to override inherited ones)
        if node_class in cls.NODE_TOOLTIPS:
            logger.info(f"Adding class's own tooltips: {cls.NODE_TOOLTIPS[node_class]}")
            tooltips.update(cls.NODE_TOOLTIPS[node_class])
        
        logger.info(f"Final tooltips for {node_class}: {tooltips}")
        return tooltips
    
    @classmethod
    def get_tooltip(cls, node_class: str, param_name: str) -> str:
        """
        Get the tooltip for a specific parameter of a node.
        
        Args:
            node_class: Name of the node class
            param_name: Name of the parameter
            
        Returns:
            str: Tooltip text for the parameter, or empty string if not found
        """
        tooltips = cls.get_tooltips(node_class)
        return tooltips.get(param_name, "")
    
    @classmethod
    def register_tooltips(cls, node_class: str, tooltips: dict, inherits_from: Optional[Union[str, List[str]]] = None):
        """
        Register tooltips for a node class.
        
        Args:
            node_class: Name of the node class
            tooltips: Dictionary mapping parameter names to tooltip descriptions
            inherits_from: Optional parent class name(s) to inherit tooltips from
        """
        logger.info(f"Registering tooltips for {node_class}")
        logger.info(f"Tooltips: {tooltips}")
        if inherits_from:
            logger.info(f"Inherits from: {inherits_from}")
            cls.INHERITANCE_MAP[node_class] = inherits_from
        cls.NODE_TOOLTIPS[node_class] = tooltips


def apply_tooltips(node_class):
    """
    Class decorator to apply tooltips to a node class.
    This will automatically add tooltips to the INPUT_TYPES configuration.
    Only applies to classes that have an INPUT_TYPES classmethod.
    
    Args:
        node_class: The node class to apply tooltips to
        
    Returns:
        The decorated node class with tooltips applied
    """
    # If the class doesn't have INPUT_TYPES, just return it unchanged
    if not hasattr(node_class, 'INPUT_TYPES'):
        logger.info(f"Class {node_class.__name__} has no INPUT_TYPES - skipping tooltip application")
        return node_class
        
    original_input_types = node_class.INPUT_TYPES
    
    @classmethod
    def input_types_with_tooltips(cls):
        input_types = original_input_types()
        tooltips = TooltipManager.get_tooltips(cls.__name__)
        logger.info(f"Getting tooltips for {cls.__name__}")
        logger.info(f"Retrieved tooltips: {tooltips}")
        
        def add_tooltip_to_config(param_name, config):
            if param_name not in tooltips:
                return config
                
            tooltip = tooltips[param_name]
            logger.info(f"Adding tooltip for {param_name}: {tooltip}")
            
            # Handle tuple format (type, config_dict)
            if isinstance(config, tuple):
                if len(config) == 2 and isinstance(config[1], dict):
                    param_type, param_config = config
                    param_config = param_config.copy()
                    param_config["tooltip"] = tooltip
                    return (param_type, param_config)
                elif len(config) == 1:
                    return (config[0], {"tooltip": tooltip})
                return config
            
            # Handle list format (dropdown options)
            if isinstance(config, list):
                return (config, {"tooltip": tooltip})
                
            # Handle direct dict format
            if isinstance(config, dict):
                config = config.copy()
                config["tooltip"] = tooltip
                return config
                
            return config
        
        # Add tooltips to required parameters
        if "required" in input_types:
            input_types["required"] = {
                param_name: add_tooltip_to_config(param_name, config)
                for param_name, config in input_types["required"].items()
            }
        
        # Add tooltips to optional parameters
        if "optional" in input_types:
            input_types["optional"] = {
                param_name: add_tooltip_to_config(param_name, config)
                for param_name, config in input_types["optional"].items()
            }
        
        logger.info(f"Final INPUT_TYPES: {input_types}")
        return input_types
    
    node_class.INPUT_TYPES = input_types_with_tooltips
    return node_class 