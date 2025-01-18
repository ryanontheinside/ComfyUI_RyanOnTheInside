import { app } from "../../scripts/app.js";

console.log("[ColorPicker] Extension loading...");

// Import iro.js from CDN
const script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/@jaames/iro@5';
document.head.appendChild(script);

script.onload = () => {
    app.registerExtension({
        name: "org.ryanontheinside.color_picker",
        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            console.log("[ColorPicker] beforeRegisterNodeDef called", { nodeType, nodeData });
            
            if (nodeData.name === "ColorPicker") {
                console.log("[ColorPicker] Found ColorPicker node, setting up...");
                
                // Set initial size with desired proportions (3:4 ratio)
                nodeType.size = [240, 320];
                
                // Maintain proportions while allowing resizing
                nodeType.prototype.computeSize = function(size) {
                    const aspectRatio = 3/4; // width:height ratio
                    if (size) {
                        // Adjust height based on width to maintain ratio
                        return [size[0], Math.floor(size[0] / aspectRatio)];
                    }
                    return [240, 320]; // Default size
                };
                
                // Override the onNodeCreated method
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                const onDrawForeground = nodeType.prototype.onDrawForeground;
                const onMouseDown = nodeType.prototype.onMouseDown;
                const onMouseMove = nodeType.prototype.onMouseMove;
                const onMouseUp = nodeType.prototype.onMouseUp;
                const onResize = nodeType.prototype.onResize;

                // Add method to calculate widget area height
                nodeType.prototype.getWidgetAreaHeight = function() {
                    let height = 0;
                    const titleHeight = 30;
                    const widgetSpacing = 4;
                    
                    height += titleHeight;
                    
                    for (const w of this.widgets) {
                        if (!w.hidden) {
                            height += w.computeSize?.[1] || 20;
                            height += widgetSpacing;
                        }
                    }
                    
                    return height + 10;  // Add padding
                };
                
                nodeType.prototype.onNodeCreated = function() {
                    console.log("[ColorPicker] onNodeCreated called");
                    const r = onNodeCreated?.apply(this, arguments);
                    
                    const margin = 30;
                    
                    // Create container for iro color picker
                    const container = document.createElement("div");
                    container.style.cssText = `
                        padding: 0;
                        box-sizing: border-box;
                        position: absolute;
                        bottom: 0;
                        left: ${margin}px;
                        right: ${margin}px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    `;
                    
                    console.log("[ColorPicker] Created container");
                    
                    let colorPicker = null;
                    
                    // Function to update outputs
                    const updateOutputs = (color) => {
                        console.log("[ColorPicker] Updating outputs with color:", color);
                        const hex = color.hexString;
                        const rgb = color.rgb;
                        const hue = color.hue;
                        
                        this.setOutputData(0, hex);
                        this.setOutputData(1, `${rgb.r},${rgb.g},${rgb.b}`);
                        this.setOutputData(2, Math.round(hue));
                    };
                    
                    // Add the DOM widget
                    const widget = this.addDOMWidget("color", "Color", container, {
                        getValue: () => colorPicker ? colorPicker.color.hexString : "#ff0000",
                        setValue: (v) => {
                            if (colorPicker) {
                                colorPicker.color.hexString = v;
                                updateOutputs(colorPicker.color);
                            }
                        },
                        serialize: true
                    });
                    
                    console.log("[ColorPicker] Added DOM widget");
                    
                    const updatePickerLayout = () => {
                        const widgetAreaHeight = this.getWidgetAreaHeight();
                        const availableHeight = Math.max(200, this.size[1] - widgetAreaHeight - margin * 2);
                        const availableWidth = this.size[0] - margin * 2;
                        const pickerSize = Math.min(availableWidth, availableHeight - 40); // Leave room for slider
                        
                        // Update container height
                        container.style.height = `${availableHeight}px`;
                        
                        return pickerSize;
                    };
                    
                    // Initialize iro color picker after widget is added to DOM
                    requestAnimationFrame(() => {
                        const pickerSize = updatePickerLayout();
                        
                        colorPicker = new iro.ColorPicker(container, {
                            width: pickerSize,
                            color: "#ff0000",
                            layout: [
                                { 
                                    component: iro.ui.Wheel,
                                    options: {}
                                },
                                {
                                    component: iro.ui.Slider,
                                    options: {
                                        sliderType: 'value',
                                        margin: 20
                                    }
                                }
                            ]
                        });
                        
                        colorPicker.on('color:change', (color) => {
                            console.log("[ColorPicker] Color changed:", color.hexString);
                            updateOutputs(color);
                        });
                        
                        widget.colorPicker = colorPicker;
                    });
                    
                    // Handle node resize
                    this.onResize = function(size) {
                        // Maintain aspect ratio
                        const newSize = this.computeSize(size);
                        size[0] = newSize[0];
                        size[1] = newSize[1];
                        
                        if (onResize) {
                            onResize.call(this, size);
                        }
                        const pickerSize = updatePickerLayout();
                        if (widget.colorPicker) {
                            widget.colorPicker.resize(pickerSize);
                        }
                        this.setDirtyCanvas(true, true);
                    };
                    
                    // Make widget serializable
                    widget.type = "color";
                    widget.options = widget.options || {};
                    widget.options.serialize = true;
                    
                    // Set initial node size
                    this.setSize([240, 320]);
                    
                    console.log("[ColorPicker] Widget setup complete");
                    
                    return r;
                };
                
                console.log("[ColorPicker] Node setup complete");
            }
        }
    });
};

console.log("[ColorPicker] Extension loaded"); 