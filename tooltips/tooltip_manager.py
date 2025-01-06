"""
Core tooltip management functionality.
"""
from typing import Dict, Optional, Union, List
from collections import deque

class TooltipManager:
    """
    Manages tooltips for all nodes in the suite.
    Provides a centralized way to define and access tooltips.
    """
    
    # Dictionary mapping node names to their parameter tooltips
    NODE_TOOLTIPS = {}
    
    # Dictionary mapping node names to their parent classes
    INHERITANCE_MAP = {}

    # Dictionary mapping node names to their descriptions
    NODE_DESCRIPTIONS = {}
    
    @classmethod
    def get_tooltips(cls, node_class: str) -> dict:
        """
        Get all tooltips for a node class, including inherited tooltips.
        Uses breadth-first search to build the inheritance chain, properly handling multiple inheritance.
        
        Args:
            node_class: Name of the node class
            
        Returns:
            Dictionary mapping parameter names to tooltip descriptions
        """
        tooltips = {}
        visited = set()
        inheritance_chain = []
        queue = deque([(node_class, 0)])  # (class_name, depth)
        max_depth = 0
        
        # Build inheritance chain using BFS, tracking depth
        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            inheritance_chain.append((current, depth))
            max_depth = max(max_depth, depth)
            
            if current in cls.INHERITANCE_MAP:
                parents = cls.INHERITANCE_MAP[current]
                if isinstance(parents, list):
                    # Add all parents at the next depth level
                    queue.extend((parent, depth + 1) for parent in parents)
                else:
                    queue.append((parents, depth + 1))
        
        # Apply tooltips by depth level, from base classes (highest depth) to derived classes
        for depth in range(max_depth, -1, -1):
            # Get all classes at this depth level
            classes_at_depth = [cls_name for cls_name, d in inheritance_chain if d == depth]
            for class_name in classes_at_depth:
                if class_name in cls.NODE_TOOLTIPS:
                    tooltips.update(cls.NODE_TOOLTIPS[class_name])
        
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
    def get_description(cls, node_class: str) -> str:
        """
        Get the description for a node class, including inherited descriptions.
        Descriptions from parent classes are combined in order.
        
        Args:
            node_class: Name of the node class
            
        Returns:
            str: Combined description text for the node
        """
        descriptions = []
        visited = set()
        queue = deque([node_class])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            
            # Add description if it exists
            if current in cls.NODE_DESCRIPTIONS:
                descriptions.append(cls.NODE_DESCRIPTIONS[current])
            
            # Add parent classes to queue
            if current in cls.INHERITANCE_MAP:
                parents = cls.INHERITANCE_MAP[current]
                if isinstance(parents, list):
                    queue.extend(parents)
                else:
                    queue.append(parents)
        
        return "\n\n".join(descriptions) if descriptions else ""
    
    @classmethod
    def register_tooltips(cls, node_class: str, tooltips: dict, inherits_from: Optional[Union[str, List[str]]] = None, description: Optional[str] = None):
        """
        Register tooltips and description for a node class.
        
        Args:
            node_class: Name of the node class
            tooltips: Dictionary mapping parameter names to tooltip descriptions
            inherits_from: Optional parent class name(s) to inherit tooltips from
            description: Optional description text for the node
        """
        if inherits_from:
            cls.INHERITANCE_MAP[node_class] = inherits_from
        cls.NODE_TOOLTIPS[node_class] = tooltips
        if description:
            cls.NODE_DESCRIPTIONS[node_class] = description


def apply_tooltips(node_class):
    """
    Class decorator to apply tooltips to a node class.
    This will automatically:
    1. Add tooltips to the INPUT_TYPES configuration
    2. Set the DESCRIPTION attribute based on registered descriptions
    Only applies to classes that have an INPUT_TYPES classmethod.
    
    Args:
        node_class: The node class to apply tooltips to
        
    Returns:
        The decorated node class with tooltips and description applied
    """
    # Set the DESCRIPTION attribute from registered descriptions
    description = TooltipManager.get_description(node_class.__name__)
    if description:
        node_class.DESCRIPTION = description
    
    # If the class doesn't have INPUT_TYPES, just return it
    if not hasattr(node_class, 'INPUT_TYPES'):
        return node_class
        
    original_input_types = node_class.INPUT_TYPES
    
    @classmethod
    def input_types_with_tooltips(cls):
        try:
            input_types = original_input_types()
            if input_types is None:
                return original_input_types.__get__(cls, cls)()
        except Exception:
            return original_input_types.__get__(cls, cls)()
            
        tooltips = TooltipManager.get_tooltips(cls.__name__)
        
        def add_tooltip_to_config(param_name, config):
            if param_name not in tooltips:
                return config
                
            tooltip = tooltips[param_name]
            
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
        
        return input_types
    
    node_class.INPUT_TYPES = input_types_with_tooltips
    return node_class 