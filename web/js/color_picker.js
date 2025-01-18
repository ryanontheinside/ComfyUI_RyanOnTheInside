import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "org.ryanontheinside.color_picker",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ColorPicker") {
            // Save original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onRemoved = nodeType.prototype.onRemoved;
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            
            // Override the onNodeCreated method
            nodeType.prototype.onNodeCreated = function() {
                console.log("ColorPicker: onNodeCreated called");
                const r = onNodeCreated?.apply(this, arguments);
                
                // Initialize internal values
                this.currentColor = "#FF0000";
                this.currentRGB = "255,0,0";
                this.currentHue = 0;
                
                // Make node resizable
                this.flags = this.flags || {};
                this.flags.resizable = true;
                
                // Set initial size - matching drawable feature approach
                this.size = [350, 450];
                
                return r;
            };
            
            // Override the onDrawForeground method
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                
                if (this.flags.collapsed) return;
                
                // Calculate widget area height
                const widgetHeight = 90;  // Height for all widgets
                const margin = 10;
                
                // Calculate picker area (remaining space after widgets and margins)
                const pickerWidth = this.size[0] - margin * 2;
                const pickerHeight = this.size[1] - widgetHeight - margin * 2;
                const pickerX = margin;
                const pickerY = widgetHeight + margin;
                
                // Draw color picker background
                ctx.fillStyle = "#2d2d2d";
                ctx.strokeStyle = "#666";
                ctx.lineWidth = 1;
                ctx.fillRect(pickerX, pickerY, pickerWidth, pickerHeight);
                ctx.strokeRect(pickerX, pickerY, pickerWidth, pickerHeight);
                
                // Draw color spectrum in top half
                const spectrumHeight = pickerHeight / 2;
                const gradient = ctx.createLinearGradient(pickerX, pickerY, pickerWidth + pickerX, pickerY);
                
                // Add rainbow colors
                gradient.addColorStop(0, "#FF0000");    // Red
                gradient.addColorStop(0.17, "#FF00FF"); // Magenta
                gradient.addColorStop(0.33, "#0000FF"); // Blue
                gradient.addColorStop(0.5, "#00FFFF");  // Cyan
                gradient.addColorStop(0.67, "#00FF00"); // Green
                gradient.addColorStop(0.83, "#FFFF00"); // Yellow
                gradient.addColorStop(1, "#FF0000");    // Red
                
                ctx.fillStyle = gradient;
                ctx.fillRect(pickerX + 2, pickerY + 2, pickerWidth - 4, spectrumHeight - 4);
                
                // Draw brightness gradient in bottom half
                const brightnessY = pickerY + spectrumHeight;
                const brightnessGradient = ctx.createLinearGradient(pickerX, brightnessY, pickerX, pickerY + pickerHeight);
                brightnessGradient.addColorStop(0, "rgba(255, 255, 255, 1)");
                brightnessGradient.addColorStop(0.5, "rgba(255, 255, 255, 0)");
                brightnessGradient.addColorStop(0.5, "rgba(0, 0, 0, 0)");
                brightnessGradient.addColorStop(1, "rgba(0, 0, 0, 1)");
                
                ctx.fillStyle = brightnessGradient;
                ctx.fillRect(pickerX + 2, brightnessY + 2, pickerWidth - 4, spectrumHeight - 4);
                
                // Draw current color indicator
                if (this.currentColor) {
                    // Draw current color preview in corner
                    ctx.fillStyle = this.currentColor;
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 2;
                    
                    // Draw a circle with the current color
                    const indicatorSize = 20;
                    const indicatorX = pickerX + pickerWidth - indicatorSize - 5;
                    const indicatorY = pickerY + pickerHeight - indicatorSize - 5;
                    
                    ctx.beginPath();
                    ctx.arc(indicatorX, indicatorY, indicatorSize/2, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                    
                    // Draw position indicator in color picker
                    if (this.currentHue !== undefined) {
                        const x = pickerX + (this.currentHue / 360) * pickerWidth;
                        let y;
                        
                        // Convert RGB to HSV to get value
                        const r = parseInt(this.currentColor.slice(1, 3), 16) / 255;
                        const g = parseInt(this.currentColor.slice(3, 5), 16) / 255;
                        const b = parseInt(this.currentColor.slice(5, 7), 16) / 255;
                        const max = Math.max(r, g, b);
                        const value = max * 100;
                        
                        if (value >= 50) {
                            // Top half
                            y = pickerY + ((100 - value) / 100) * (pickerHeight / 2);
                        } else {
                            // Bottom half
                            y = pickerY + pickerHeight/2 + ((50 - value) / 50) * (pickerHeight / 2);
                        }
                        
                        // Draw position indicator
                        ctx.beginPath();
                        ctx.arc(x, y, 5, 0, Math.PI * 2);
                        ctx.strokeStyle = value > 50 ? "#000" : "#fff";
                        ctx.lineWidth = 2;
                        ctx.stroke();
                        ctx.fillStyle = value > 50 ? "#fff" : "#000";
                        ctx.fill();
                    }
                }
                
                // Store click area for later
                this.colorPickerArea = {
                    x: pickerX,
                    y: pickerY,
                    width: pickerWidth,
                    height: pickerHeight
                };
            };
            
            // Add click handler
            nodeType.prototype.onMouseDown = function(e, pos, graphcanvas) {
                if (this.flags.collapsed) return false;
                
                // Get canvas coordinates like in drawable_feature.js
                const scale = graphcanvas.ds.scale;
                const offset = graphcanvas.ds.offset;
                
                // Calculate node position in canvas space
                const nodeX = this.pos[0] * scale + offset[0];
                const nodeY = this.pos[1] * scale + offset[1];
                
                // Get mouse position relative to canvas
                const rect = graphcanvas.canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                
                // Calculate click position relative to node
                const localX = (mouseX - nodeX) / scale;
                const localY = (mouseY - nodeY) / scale;
                
                console.log("=== Color Picker Click Debug ===");
                console.log("Scale:", scale);
                console.log("Offset:", offset);
                console.log("Node pos:", this.pos);
                console.log("Node canvas pos:", nodeX, nodeY);
                console.log("Mouse pos:", mouseX, mouseY);
                console.log("Local pos:", localX, localY);
                console.log("Picker area:", this.colorPickerArea);
                
                if (this.colorPickerArea) {
                    const { x: px, y: py, width, height } = this.colorPickerArea;
                    
                    // Check if click is within picker area
                    if (localX >= px && localX <= px + width && localY >= py && localY <= py + height) {
                        // Calculate relative position within color picker
                        const relX = (localX - px) / width;
                        const relY = (localY - py) / height;
                        
                        console.log("Click in picker area!");
                        console.log("Relative position:", relX, relY);
                        
                        // Get base hue from X position
                        const hue = Math.max(0, Math.min(360, relX * 360));
                        
                        // Get saturation and value based on Y position
                        let saturation = 100;
                        let value = 100;
                        
                        if (relY <= 0.5) {
                            // Top half - full saturation, value varies from 100 to 50
                            value = 100 - (relY * 2 * 100);
                        } else {
                            // Bottom half - value varies from 50 to 0
                            value = 50 - ((relY - 0.5) * 2 * 100);
                        }
                        
                        value = Math.max(0, Math.min(100, value));
                        
                        console.log("HSV values:", { hue, saturation, value });
                        
                        // Convert HSV to RGB
                        const rgb = this.hsvToRgb(hue, saturation, value);
                        const hexColor = '#' + rgb.map(x => {
                            const hex = Math.round(Math.max(0, Math.min(255, x))).toString(16);
                            return hex.length === 1 ? '0' + hex : hex;
                        }).join('').toUpperCase();
                        
                        console.log("Final color:", hexColor);
                        
                        // Update color
                        this.updateColor(hexColor);
                        
                        // Trigger widget callbacks
                        const colorWidget = this.widgets.find(w => w.name === "color");
                        const rgbWidget = this.widgets.find(w => w.name === "rgb_value");
                        const hueWidget = this.widgets.find(w => w.name === "hue");
                        
                        if (colorWidget?.callback) colorWidget.callback(hexColor);
                        if (rgbWidget?.callback) rgbWidget.callback(`${Math.round(rgb[0])},${Math.round(rgb[1])},${Math.round(rgb[2])}`);
                        if (hueWidget?.callback) hueWidget.callback(hue);
                        
                        return true;
                    } else {
                        console.log("Click outside picker area");
                    }
                }
                
                return false;
            };
            
            // Add HSV to RGB conversion
            nodeType.prototype.hsvToRgb = function(h, s, v) {
                s = s / 100;
                v = v / 100;
                
                const i = Math.floor(h / 60);
                const f = h / 60 - i;
                const p = v * (1 - s);
                const q = v * (1 - f * s);
                const t = v * (1 - (1 - f) * s);
                
                let r, g, b;
                switch (i % 6) {
                    case 0: r = v; g = t; b = p; break;
                    case 1: r = q; g = v; b = p; break;
                    case 2: r = p; g = v; b = t; break;
                    case 3: r = p; g = q; b = v; break;
                    case 4: r = t; g = p; b = v; break;
                    case 5: r = v; g = p; b = q; break;
                }
                
                return [r * 255, g * 255, b * 255];
            };
            
            // Update color values
            nodeType.prototype.updateColor = function(hexColor) {
                console.log("ColorPicker: Updating color values for", hexColor);
                
                // Store hex color
                this.currentColor = hexColor.toUpperCase();
                
                // Convert hex to RGB
                const r = parseInt(hexColor.slice(1, 3), 16);
                const g = parseInt(hexColor.slice(3, 5), 16);
                const b = parseInt(hexColor.slice(5, 7), 16);
                
                // Store RGB
                this.currentRGB = `${r},${g},${b}`;
                
                // Convert RGB to HSV to get hue
                const max = Math.max(r, g, b);
                const min = Math.min(r, g, b);
                let h = 0;
                
                if (max === min) {
                    h = 0;
                } else {
                    const d = max - min;
                    switch (max) {
                        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                        case g: h = (b - r) / d + 2; break;
                        case b: h = (r - g) / d + 4; break;
                    }
                    h *= 60;
                }
                
                // Store hue as integer
                this.currentHue = Math.round(h);
                
                // Update hidden widgets
                const widgets = this.widgets || [];
                for (const w of widgets) {
                    if (w.name === "color") w.value = this.currentColor;
                    if (w.name === "rgb_value") w.value = this.currentRGB;
                    if (w.name === "hue") w.value = this.currentHue;
                }
                
                // Force redraw
                this.setDirtyCanvas(true, true);
                
                // Update outputs
                this.setOutputData(0, this.currentColor);
                this.setOutputData(1, this.currentRGB);
                this.setOutputData(2, this.currentHue);
            };
        }
    }
}); 