"""Generate tooltip stubs for node classes"""
import os
from pathlib import Path

def generate_tooltip_file(category: str, node_classes: list) -> str:
    """Generate tooltip file content"""
    lines = [
        f'"""Tooltips for {category}-related nodes."""',
        "",
        "from ..tooltip_manager import TooltipManager",
        "",
        "def register_tooltips():",
        f'    """Register tooltips for {category} nodes"""',
        ""
    ]
    
    for node_class in node_classes:
        class_name = node_class["name"]
        bases = node_class["bases"]
        
        # Add class comment
        if bases:
            lines.append(f"    # {class_name} tooltips (inherits from: {', '.join(bases)})")
        else:
            lines.append(f"    # {class_name} tooltips")
            
        # Start tooltip registration
        lines.append(f'    TooltipManager.register_tooltips("{class_name}", {{')
        lines.append('        # TODO: Add parameter tooltips')
        
        # Close registration
        if bases:
            # For multiple inheritance, pass all bases
            if len(bases) > 1:
                lines.append(f"    }}, inherits_from={repr(bases)})")
            else:
                lines.append(f"    }}, inherits_from='{bases[0]}')")
        else:
            lines.append("    })")
        lines.append("")
    
    return "\n".join(lines)

def main():
    """Main function"""
    # Create output directories
    tooltips_dir = Path("tooltips")
    categories_dir = tooltips_dir / "categories"
    categories_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directories: {tooltips_dir}, {categories_dir}")
    
    # Create __init__.py files
    (tooltips_dir / "__init__.py").write_text('"""Tooltip system"""\n')
    (categories_dir / "__init__.py").write_text('"""Tooltip categories"""\n')
    
    # Process each category directory
    nodes_dir = Path("nodes")
    print(f"Looking for nodes in: {nodes_dir}")
    for category_dir in nodes_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
            
        category = category_dir.name
        print(f"\nProcessing category: {category}")
        node_classes = []
        
        # Find all Python files
        for py_file in category_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            print(f"  Processing file: {py_file.name}")
            # Read file content
            content = py_file.read_text()
            
            # Find class definitions
            in_class = False
            class_name = None
            bases = []
            
            for line in content.split("\n"):
                if line.strip().startswith("class "):
                    class_def = line.strip().replace("class ", "").replace(":", "")
                    if "(" in class_def:
                        class_name = class_def.split("(")[0].strip()
                        bases = [b.strip() for b in class_def.split("(")[1].rstrip(")").split(",")]
                        # Clean up any empty bases
                        bases = [b for b in bases if b]
                    else:
                        class_name = class_def.strip()
                        bases = []
                    print(f"    Found class: {class_name} (bases: {bases})")
                    node_classes.append({
                        "name": class_name,
                        "bases": bases
                    })
                        
        if node_classes:
            # Generate and write tooltip file
            tooltip_file = categories_dir / f"{category}.py"
            tooltip_file.write_text(generate_tooltip_file(category, node_classes))
            print(f"Generated {tooltip_file} with {len(node_classes)} classes")

if __name__ == "__main__":
    main() 