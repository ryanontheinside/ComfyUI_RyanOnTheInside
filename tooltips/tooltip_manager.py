"""
Core tooltip management functionality.
"""
from typing import Dict, Optional, Union, List
from collections import deque
import logging

# Configure logging
logger = logging.getLogger('tooltip_manager')
logger.setLevel(logging.DEBUG)

# Create handlers if no handlers exist
if not logger.handlers:
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler('tooltips.log')
    fh.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

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
        logger.info(f"Getting tooltips for node class: {node_class}")
        tooltips = {}
        visited = set()
        inheritance_chain = []
        queue = deque([(node_class, 0)])  # (class_name, depth)
        max_depth = 0
        
        logger.debug(f"Starting BFS for inheritance chain of {node_class}")
        # Build inheritance chain using BFS, tracking depth
        while queue:
            current, depth = queue.popleft()
            if current in visited:
                logger.debug(f"Skipping already visited class: {current}")
                continue
                
            visited.add(current)
            inheritance_chain.append((current, depth))
            max_depth = max(max_depth, depth)
            logger.debug(f"Processing class {current} at depth {depth}")
            
            if current in cls.INHERITANCE_MAP:
                parents = cls.INHERITANCE_MAP[current]
                logger.debug(f"Found parents for {current}: {parents}")
                if isinstance(parents, list):
                    queue.extend((parent, depth + 1) for parent in parents)
                else:
                    queue.append((parents, depth + 1))
        
        logger.debug(f"Inheritance chain for {node_class}: {inheritance_chain}")
        
        # Process tooltips in natural order (depth 0 to max_depth)
        # This means we'll process derived classes first, which is what we want
        # since derived class tooltips should override base class tooltips
        for depth in range(0, max_depth + 1):
            # Get all classes at this depth level
            classes_at_depth = [cls_name for cls_name, d in inheritance_chain if d == depth]
            logger.debug(f"Processing classes at depth {depth}: {classes_at_depth}")
            for class_name in classes_at_depth:
                if class_name in cls.NODE_TOOLTIPS:
                    # Add tooltips from this class, but don't override any existing tooltips
                    # This ensures derived class tooltips take precedence over base class tooltips
                    class_tooltips = cls.NODE_TOOLTIPS[class_name]
                    logger.debug(f"Adding tooltips from {class_name}: {list(class_tooltips.keys())}")
                    for param, tooltip in class_tooltips.items():
                        if param not in tooltips:  # Only add if not already defined
                            tooltips[param] = tooltip
                            logger.debug(f"Added tooltip for {param} from {class_name}")
        
        logger.info(f"Final tooltips for {node_class}: {list(tooltips.keys())}")
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
        logger.debug(f"Getting tooltip for {node_class}.{param_name}")
        tooltips = cls.get_tooltips(node_class)
        result = tooltips.get(param_name, "")
        logger.debug(f"Tooltip for {node_class}.{param_name}: {'<empty>' if not result else 'found'}")
        return result

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
        logger.info(f"Getting description for node class: {node_class}")
        descriptions = []
        visited = set()
        queue = deque([node_class])

        while queue:
            current = queue.popleft()
            if current in visited:
                logger.debug(f"Skipping already visited class for description: {current}")
                continue
                
            visited.add(current)
            
            # Add description if it exists
            if current in cls.NODE_DESCRIPTIONS:
                logger.debug(f"Found description for {current}")
                descriptions.append(cls.NODE_DESCRIPTIONS[current])
            
            # Add parent classes to queue
            if current in cls.INHERITANCE_MAP:
                parents = cls.INHERITANCE_MAP[current]
                logger.debug(f"Adding parent classes for description lookup: {parents}")
                if isinstance(parents, list):
                    queue.extend(parents)
                else:
                    queue.append(parents)
        
        result = "\n\n".join(descriptions) if descriptions else ""
        logger.info(f"Final description for {node_class}: {'<empty>' if not result else 'found'}")
        return result
    
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
        logger.info(f"Registering tooltips for {node_class}")
        logger.debug(f"Tooltip params being registered: {list(tooltips.keys())}")
        
        if inherits_from:
            logger.debug(f"Setting inheritance for {node_class}: {inherits_from}")
            cls.INHERITANCE_MAP[node_class] = inherits_from
        cls.NODE_TOOLTIPS[node_class] = tooltips
        if description:
            logger.debug(f"Setting description for {node_class}")
            cls.NODE_DESCRIPTIONS[node_class] = description


def apply_tooltips(node_class):
    """
    Class decorator to apply tooltips to a node class.
    This will automatically:
    1. Add tooltips to the INPUT_TYPES configuration
    2. Set the DESCRIPTION attribute based on registered descriptions
    Only applies to classes that have an INPUT_TYPES classmethod.
    """
    logger.info(f"Applying tooltips decorator to class: {node_class.__name__}")
    
    # Set the DESCRIPTION attribute from registered descriptions
    description = TooltipManager.get_description(node_class.__name__)
    if description:
        logger.debug(f"Setting DESCRIPTION for {node_class.__name__}")
        node_class.DESCRIPTION = description
    
    # If the class doesn't have INPUT_TYPES, just return it
    if not hasattr(node_class, 'INPUT_TYPES'):
        logger.warning(f"Class {node_class.__name__} has no INPUT_TYPES, skipping tooltip application")
        return node_class
        
    original_input_types = node_class.INPUT_TYPES
    
    @classmethod
    def input_types_with_tooltips(cls):
        """
        Wrapper for INPUT_TYPES that adds tooltips to the configuration.
        Handles tooltip application failures gracefully.
        """
        logger.debug(f"Getting input types with tooltips for {cls.__name__}")
        
        # Get the original method result first - this must succeed for the node to work
        try:
            input_types = original_input_types.__get__(cls, cls)()
            logger.debug(f"Original input types for {cls.__name__}: {list(input_types.keys()) if isinstance(input_types, dict) else 'non-dict'}")
        except Exception as e:
            logger.error(f"Failed to get original input types for {cls.__name__}: {str(e)}", exc_info=True)
            raise
            
        # If not a dict or empty, return as-is
        if not isinstance(input_types, dict) or not input_types:
            logger.warning(f"Input types for {cls.__name__} is not a dict or is empty, returning as-is")
            return input_types
            
        # Now try to add tooltips - any failure here should be contained
        try:
            tooltips = TooltipManager.get_tooltips(cls.__name__)
            logger.debug(f"Retrieved tooltips for {cls.__name__}: {list(tooltips.keys())}")
            
            def add_tooltip_to_config(param_name, config):
                if param_name not in tooltips:
                    logger.debug(f"No tooltip found for parameter {param_name}")
                    return config
                    
                try:
                    tooltip = tooltips[param_name]
                    logger.debug(f"Adding tooltip to {param_name}: {tooltip[:50]}...")
                    
                    # Handle tuple format (type, config_dict)
                    if isinstance(config, tuple) and len(config) == 2:
                        param_type, param_config = config
                        # If param_config is already a dict, just add the tooltip
                        if isinstance(param_config, dict):
                            param_config = param_config.copy()
                            param_config["tooltip"] = tooltip
                            return (param_type, param_config)
                        # If param_type is a list, this is a dropdown without config
                        elif isinstance(param_type, list):
                            return (param_type, {"tooltip": tooltip})
                        # If param_type is a tuple containing a method call result, preserve it
                        elif isinstance(param_type, tuple) and len(param_type) > 0:
                            return (param_type, {"tooltip": tooltip})
                    logger.debug(f"Could not add tooltip to {param_name} - unsupported config format")
                    return config
                except Exception as e:
                    logger.error(f"Failed to add tooltip to {param_name}: {str(e)}", exc_info=True)
                    return config
            
            # Process required parameters
            if "required" in input_types:
                logger.debug(f"Processing required parameters for {cls.__name__}")
                input_types["required"] = {
                    name: add_tooltip_to_config(name, config)
                    for name, config in input_types["required"].items()
                }
            
            # Process optional parameters
            if "optional" in input_types:
                logger.debug(f"Processing optional parameters for {cls.__name__}")
                input_types["optional"] = {
                    name: add_tooltip_to_config(name, config)
                    for name, config in input_types["optional"].items()
                }
        except Exception as e:
            logger.error(f"Failed to process tooltips for {cls.__name__}: {str(e)}", exc_info=True)
            
        return input_types
    
    # Replace the original INPUT_TYPES with our wrapped version
    logger.info(f"Replacing INPUT_TYPES for {node_class.__name__} with tooltip-enabled version")
    node_class.INPUT_TYPES = input_types_with_tooltips
    
    return node_class 