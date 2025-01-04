"""
Script to fill in tooltips from node_configs.py
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, Any

def parse_node_configs(config_file: str) -> Dict[str, Dict[str, str]]:
    """Parse node_configs.py to extract parameter descriptions"""
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all node config blocks
    config_blocks = re.finditer(
        r'add_node_config\(\s*["\']([^"\']+)["\']\s*,\s*{([^}]+)}\s*\)',
        content, re.DOTALL
    )
    
    node_descriptions = {}
    for match in config_blocks:
        node_name = match.group(1)
        config_block = match.group(2)
        
        # Extract parameter descriptions from BASE_DESCRIPTION and ADDITIONAL_INFO
        param_descriptions = {}
        
        # Parse BASE_DESCRIPTION
        base_desc_match = re.search(
            r'BASE_DESCRIPTION\s*:\s*["\']([^"\']+)["\']',
            config_block
        )
        if base_desc_match:
            param_descriptions['_base'] = base_desc_match.group(1)
        
        # Parse ADDITIONAL_INFO
        additional_info_match = re.search(
            r'ADDITIONAL_INFO\s*:\s*{([^}]+)}',
            config_block, re.DOTALL
        )
        if additional_info_match:
            info_block = additional_info_match.group(1)
            param_matches = re.finditer(
                r'["\']([^"\']+)["\']\s*:\s*["\']([^"\']+)["\']',
                info_block
            )
            for param_match in param_matches:
                param_name = param_match.group(1)
                param_desc = param_match.group(2)
                param_descriptions[param_name] = param_desc
        
        node_descriptions[node_name] = param_descriptions
    
    return node_descriptions

def update_tooltip_file(tooltip_file: str, node_descriptions: Dict[str, Dict[str, str]]) -> None:
    """Update a tooltip category file with descriptions from node_configs"""
    with open(tooltip_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all tooltip registration blocks
    tooltip_blocks = re.finditer(
        r'TooltipManager\.register_tooltips\(\s*["\']([^"\']+)["\']\s*,\s*{([^}]+)}\s*(?:,\s*inherits_from=[^)]+)?\)',
        content, re.DOTALL
    )
    
    updated_content = content
    for match in tooltip_blocks:
        node_name = match.group(1)
        tooltip_block = match.group(2)
        
        if node_name not in node_descriptions:
            continue
        
        descriptions = node_descriptions[node_name]
        
        # Update each parameter's tooltip
        param_matches = re.finditer(
            r'["\']([^"\']+)["\']\s*:\s*["\']([^"\']*)["\']',
            tooltip_block
        )
        
        new_tooltips = {}
        for param_match in param_matches:
            param_name = param_match.group(1)
            if param_name in descriptions:
                new_tooltips[param_name] = descriptions[param_name]
            elif '_base' in descriptions:
                # Use base description as fallback
                new_tooltips[param_name] = descriptions['_base']
        
        # Format new tooltip block
        new_block = '{\n'
        for param, desc in new_tooltips.items():
            new_block += f'        "{param}": "{desc}",\n'
        new_block = new_block.rstrip(',\n') + '\n    }'
        
        # Replace old tooltip block with new one
        start_idx = match.start(2)
        end_idx = match.end(2)
        updated_content = (
            updated_content[:start_idx] +
            new_block +
            updated_content[end_idx:]
        )
    
    with open(tooltip_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)

def main() -> None:
    """Main function to fill tooltips from node configs"""
    # Get workspace root
    workspace_root = Path(os.getcwd())
    
    # Parse node configs
    config_file = workspace_root / 'node_configs' / 'node_configs.py'
    node_descriptions = parse_node_configs(str(config_file))
    
    # Update each tooltip category file
    tooltips_dir = workspace_root / 'tooltips' / 'categories'
    for tooltip_file in tooltips_dir.glob('*.py'):
        if tooltip_file.name != '__init__.py':
            update_tooltip_file(str(tooltip_file), node_descriptions)
            print(f'Updated tooltips in {tooltip_file}')

if __name__ == '__main__':
    main() 