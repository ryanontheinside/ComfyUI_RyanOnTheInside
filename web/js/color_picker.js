import { app } from "../../scripts/app.js";

class ColorPickerWidget {
    constructor(node) {
        this.node = node;
        
        // Create container element
        this.element = document.createElement('div');
        this.element.style.width = '200px';
        this.element.style.height = '300px';
        this.element.style.position = 'relative';
        this.element.style.backgroundColor = '#1a1a1a';
        this.element.style.borderRadius = '8px';
        this.element.style.padding = '10px';
        
        // Create color picker container
        this.pickerContainer = document.createElement('div');
        this.pickerContainer.style.width = '100%';
        this.pickerContainer.style.height = '100%';
        this.element.appendChild(this.pickerContainer);
        
        // Initialize color picker once iro.js is loaded
        this.initColorPicker();
    }
    
    initColorPicker() {
        if (window.iro) {
            // Get initial color from node's state
            const initialColor = this.node.color || "#ff0000";
            
            // Create color picker
            this.colorPicker = new window.iro.ColorPicker(this.pickerContainer, {
                width: 180,
                color: initialColor,
                layout: [
                    { 
                        component: iro.ui.Wheel,
                        options: {}
                    },
                    {
                        component: iro.ui.Slider,
                        options: {
                            sliderType: 'value'
                        }
                    }
                ]
            });
            
            // Add color change handler
            this.colorPicker.on('color:change', (color) => {
                // Update node state
                this.node.color = color.hexString;
                this.node.rgb = `${color.rgb.r},${color.rgb.g},${color.rgb.b}`;
                this.node.hue = Math.round(color.hue);
                
                // Update hidden widget for Python
                if (this.node.widgets) {
                    const colorWidget = this.node.widgets.find(w => w.name === "color");
                    if (colorWidget) {
                        colorWidget.value = JSON.stringify({
                            hex: this.node.color,
                            rgb: this.node.rgb,
                            hue: this.node.hue
                        });
                        // Trigger widget change using the node's method
                        if (this.node.onWidgetChanged) {
                            this.node.onWidgetChanged("color", colorWidget.value, colorWidget.value, colorWidget);
                        }
                    }
                }
                
                // Request canvas update
                this.node.setDirtyCanvas(true, true);
            });
        } else {
            // Retry in 100ms if iro.js hasn't loaded yet
            setTimeout(() => this.initColorPicker(), 100);
        }
    }
    
    cleanup() {
        if (this.colorPicker) {
            // Remove event listeners
            this.colorPicker.off('color:change');
            // Destroy picker instance
            this.colorPicker = null;
        }
        // Remove DOM elements
        this.element.remove();
    }
}

app.registerExtension({
    name: "RyanOnTheInside.ColorPicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ColorPicker") {
            // Store original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onSerialize = nodeType.prototype.onSerialize;
            const onConfigure = nodeType.prototype.onConfigure;
            
            // Add serialization support
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) {
                    onSerialize.apply(this, arguments);
                }
                o.color = this.color;
                o.rgb = this.rgb;
                o.hue = this.hue;
            };

            // Add deserialization support
            nodeType.prototype.onConfigure = function(o) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                this.color = o.color || "#ff0000";
                this.rgb = o.rgb || "255,0,0";
                this.hue = o.hue || 0;
                
                // Update widget with loaded values
                if (this.widgets) {
                    const colorWidget = this.widgets.find(w => w.name === "color");
                    if (colorWidget) {
                        colorWidget.value = JSON.stringify({
                            hex: this.color,
                            rgb: this.rgb,
                            hue: this.hue
                        });
                    }
                }

                // Update color picker visual state if it exists
                if (this.colorPickerWidget && this.colorPickerWidget.colorPicker) {
                    this.colorPickerWidget.colorPicker.color.set(this.color);
                }
            };
            
            // Override onNodeCreated
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply(this, arguments);
                
                // Set initial node size
                this.setSize([220, 380]);
                
                // Initialize state
                this.color = "#ff0000";
                this.rgb = "255,0,0";
                this.hue = 0;
                
                // Initialize widgets array if it doesn't exist
                if (!this.widgets) {
                    this.widgets = [];
                }
                
                // Remove default color widget if it exists
                const colorWidgetIndex = this.widgets.findIndex(w => w.name === "color");
                if (colorWidgetIndex > -1) {
                    this.widgets.splice(colorWidgetIndex, 1);
                }
                
                // Create and add the color picker widget
                const colorPickerWidget = new ColorPickerWidget(this);
                this.colorPicker = this.addDOMWidget("colorpicker", "ColorPicker", colorPickerWidget.element, {
                    serialize: false,
                    hideOnZoom: false
                });
                
                // Store widget instance for cleanup
                this.colorPickerWidget = colorPickerWidget;

                // Add hidden color widget at the end
                this.widgets.push({
                    type: "text",
                    name: "color",
                    value: JSON.stringify({
                        hex: this.color,
                        rgb: this.rgb,
                        hue: this.hue
                    }),
                    hidden: true
                });
                
                return r;
            };
            
            // Add cleanup on node removal
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (this.colorPickerWidget) {
                    this.colorPickerWidget.cleanup();
                }
                return onRemoved?.apply(this, arguments);
            };
        }
    }
});
