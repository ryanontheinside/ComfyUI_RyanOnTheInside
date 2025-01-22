import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { FeatureTemplates } from "./feature_templates.js";


//TODO: add handling for maximum frame count change
// Register custom widget type
app.registerExtension({
    name: "RyanOnTheInside.DrawableFeature",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // console.log("Registering DrawableFeature extension", { nodeType, nodeData });
        
        if (nodeData.name === "DrawableFeatureNode") {
            // console.log("Found DrawableFeatureNode, setting up widget");
            
            // Set default size
            nodeType.size = [700, 800];  // Increased size for better usability
            
            // Store the original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            const onMouseDown = nodeType.prototype.onMouseDown;
            const onMouseMove = nodeType.prototype.onMouseMove;
            const onMouseUp = nodeType.prototype.onMouseUp;
            const onDblClick = nodeType.prototype.onDblClick;
            const onResize = nodeType.prototype.onResize;
            const onSerialize = nodeType.prototype.onSerialize;
            const onConfigure = nodeType.prototype.onConfigure;
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            
            // Add serialization support
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) {
                    onSerialize.apply(this, arguments);
                }
                o.points = this.points;
            };

            // Add deserialization support
            nodeType.prototype.onConfigure = function(o) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                if (o.points) {
                    this.points = o.points;
                    this.updatePointsValue();
                }
            };
            
            // Add connection change handler
            nodeType.prototype.onConnectionsChange = function(slotType, slot, isConnected, link_info, output) {
                // console.log("onConnectionsChange:", { slotType, slot, isConnected, link_info });
                
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                
                // Handle frame_count input connection changes
                if (slotType === LiteGraph.INPUT) {
                    const input = this.inputs[slot];
                    // console.log("Input connection change:", { input, name: input?.name });
                    
                    if (input && input.name === "frame_count") {
                        if (isConnected && link_info) {
                            // When connected, we'll need to listen for value changes
                            const inputNode = link_info.origin_id ? this.graph._nodes_by_id[link_info.origin_id] : null;
                            // console.log("Connected node:", inputNode);
                            
                            if (inputNode) {
                                // Store the connected node for later reference
                                this.frameCountInputNode = inputNode;
                                // Set up the trigger and mode for value changes
                                input.onTrigger = this.onInputUpdated.bind(this);
                                // Set mode to trigger on any value change
                                input.mode = LiteGraph.UP | LiteGraph.DOWN | LiteGraph.EVENT;
                                // Also watch the connected node's widget for changes
                                const widget = inputNode.widgets?.[0];
                                if (widget) {
                                    const originalCallback = widget.callback;
                                    widget.callback = (v) => {
                                        if (originalCallback) {
                                            originalCallback.call(widget, v);
                                        }
                                        // Trigger our input update when the upstream widget changes
                                        console.log("Frame count input updated from upstream widget:", v);
                                        this.onInputUpdated();
                                    };
                                }
                                // Trigger initial update
                                this.onInputUpdated();
                            }
                        } else {
                            // When disconnected, clear the reference and trigger
                            this.frameCountInputNode = null;
                            input.onTrigger = null;
                            input.mode = LiteGraph.DEFAULT;  // Reset mode
                        }
                        // Redraw since frame count may have changed
                        this.setDirtyCanvas(true, true);
                    }
                }
            };

            // Add handler for input value changes
            nodeType.prototype.onInputUpdated = function() {
                // Find frame_count input
                const frameCountInput = this.inputs.find(input => input.name === "frame_count");
                
                if (frameCountInput && frameCountInput.link !== null) {
                    // Get the connected node
                    const link = this.graph.links[frameCountInput.link];
                    
                    if (link && link.origin_id) {
                        const inputNode = this.graph._nodes_by_id[link.origin_id];
                        
                        if (inputNode) {
                            // Try to get value from node's widget first
                            const widget = inputNode.widgets?.[0];
                            let frameCount;
                            if (widget) {
                                frameCount = widget.value;
                            } else {
                                // Fallback to output value if no widget
                                frameCount = inputNode.getOutputData(link.origin_slot);
                            }
                            
                            if (frameCount !== undefined && frameCount !== null) {
                                // Get frame count widget
                                const frameCountWidget = this.widgets.find(w => w.name === "frame_count");
                                if (frameCountWidget) {
                                    const oldFrameCount = frameCountWidget.value;
                                    // console.log("Updating frame count from", oldFrameCount, "to", frameCount);
                                    
                                    // Update widget value first
                                    frameCountWidget.value = frameCount;
                                    
                                    // Scale points if needed
                                    if (this.points && this.points.length > 0 && oldFrameCount !== frameCount) {
                                        this.points = this.points.map(([frame, value]) => {
                                            const oldMax = oldFrameCount - 1;
                                            const newMax = frameCount - 1;
                                            const newFrame = Math.min(newMax, Math.round((frame / oldMax) * newMax));
                                            return [newFrame, value];
                                        });
                                        this.points.sort((a, b) => a[0] - b[0]);
                                        this.updatePointsValue();
                                    }
                                    
                                    // Force graph area update
                                    this.setSize(this.size); // This will trigger size update and redraw
                                    this.setDirtyCanvas(true, true);
                                }
                            }
                        }
                    }
                }
            };
            
            // Override onNodeCreated to initialize the node
            nodeType.prototype.onNodeCreated = function() {
                // console.log("Node created, initializing node");
                const r = onNodeCreated?.apply(this, arguments);
                
                // Add trigger for input updates
                this.onInputTriggered = this.onInputUpdated.bind(this);
                
                // Register for input triggers
                if (!this.inputs) {
                    this.inputs = [];
                }
                
                // Find or create frame_count input
                let frameCountInput = this.inputs.find(input => input.name === "frame_count");
                if (frameCountInput) {
                    // Add trigger callback
                    frameCountInput.onTrigger = this.onInputTriggered;
                    // Set mode to trigger on value change
                    frameCountInput.mode = LiteGraph.UP | LiteGraph.DOWN;
                }
                
                // Remove default points widget if it exists
                const pointsWidgetIndex = this.widgets.findIndex(w => w.name === "points");
                if (pointsWidgetIndex > -1) {
                    this.widgets.splice(pointsWidgetIndex, 1);
                }
                
                // Initialize state
                this.points = [];
                this.isDragging = false;
                this.selectedPoint = null;
                this.hoverPoint = null;
                this.isAddingPoint = false;
                
                // Add frame count change handler
                const frameCountWidget = this.widgets.find(w => w.name === "frame_count");
                if (frameCountWidget) {
                    // Track the last frame count outside the callback
                    let lastFrameCount = frameCountWidget.value;
                    const originalCallback = frameCountWidget.callback;
                    frameCountWidget.callback = function(v, e, skipCallback) {
                        // console.log("Frame count widget callback triggered");
                        const oldFrameCount = lastFrameCount;  // Use tracked value
                        // console.log("Current frame count:", oldFrameCount);
                        // console.log("New frame count:", v);
                        // console.log("this.points exists:", !!this.points);
                        // console.log("this.points:", JSON.stringify(this.points));
                        // console.log("this.points length:", this.points?.length);
                        // console.log("oldFrameCount !== v:", oldFrameCount !== v);

                        // Update tracked value before calling original callback
                        lastFrameCount = v;
                        
                        // Call original callback to update the widget value
                        if (originalCallback && !skipCallback) {
                            originalCallback.call(frameCountWidget, v, e);
                        }
                        
                        // Now scale points based on the frame count change
                        if (this.points && this.points.length > 0 && oldFrameCount !== v) {
                            // console.log("Original points:", JSON.stringify(this.points));
                            this.points = this.points.map(([frame, value]) => {
                                const oldMax = oldFrameCount - 1;  // Convert to 0-based
                                const newMax = v - 1;  // Convert to 0-based
                                const newFrame = Math.min(newMax, Math.round((frame / oldMax) * newMax));
                                // console.log(`Scaling point frame ${frame} to ${newFrame} (old max: ${oldMax}, new max: ${newMax})`);
                                return [newFrame, value];
                            });
                            // Sort points by frame number
                            this.points.sort((a, b) => a[0] - b[0]);
                            // console.log("Scaled points:", JSON.stringify(this.points));
                            this.updatePointsValue();
                            this.setDirtyCanvas(true, true);
                        } else {
                            // console.log("Skipped points scaling because:", {
                            //     hasPoints: !!this.points,
                            //     pointsLength: this.points?.length,
                            //     frameCountChanged: oldFrameCount !== v
                            // });
                        }
                    }.bind(this);  // Still bind to node instance for this.points access
                }
                
                // Add hidden points widget at the end
                this.widgets.push({
                    type: "text",
                    name: "points",
                    value: "[]",
                    hidden: true
                });
                
                // Restore points from widget value
                try {
                    const savedPoints = JSON.parse(this.widgets[this.widgets.length - 1].value);
                    if (Array.isArray(savedPoints)) {
                        this.points = savedPoints;
                    }
                } catch (e) {
                    // console.error("Failed to restore points:", e);
                }
                

                // Add template dropdown and controls
                const templateWidget = this.addWidget("combo", "Template", "selected_template", (v) => {}, { 
                    values: [
                        "none",
                        // Basic waveforms
                        "sine",
                        "square",
                        "triangle",
                        "sawtooth",
                        // Animation curves
                        "easeInOut",
                        "easeIn",
                        "easeOut",
                        // Special patterns
                        "bounce",
                        "pulse",
                        "random",
                        "smoothRandom",
                        "heartbeat",
                        "steps"
                    ],
                    value: "none"
                });
                templateWidget.name = "template";

                // Add cycles control
                const cyclesWidget = this.addWidget("number", "Template Cycles", "template_cycles", function(v) {
                    return Math.max(0.1, Math.min(200, v));
                }, { 
                    value: 1.0,
                    min: 0.1,
                    max: 200,
                    step: 0.1,
                    precision: 2
                });
                cyclesWidget.name = "template_cycles";
                cyclesWidget.value = 1.0; // Explicitly set initial value

                // Add load template button
                const loadButton = this.addWidget("button", "Load Template", "load", () => {
                    const templateWidget = this.widgets.find(w => w.name === "template");
                    if (templateWidget && templateWidget.value && templateWidget.value !== "none") {
                        this.loadTemplate(templateWidget.value);
                    }
                });


                // Add clear button widget
                this.addWidget("button", "Clear Graph", "clear", () => {
                    this.points = [];
                    this.updatePointsValue();
                    this.setDirtyCanvas(true, true);
                });

                // Add handlers for min/max value changes
                const minValueWidget = this.widgets.find(w => w.name === "min_value");
                const maxValueWidget = this.widgets.find(w => w.name === "max_value");
                
                if (minValueWidget) {
                    const originalCallback = minValueWidget.callback;
                    minValueWidget.callback = (v) => {
                        const result = originalCallback?.call(this, v);
                        this.clampPoints();
                        return result;
                    };
                }
                
                if (maxValueWidget) {
                    const originalCallback = maxValueWidget.callback;
                    maxValueWidget.callback = (v) => {
                        const result = originalCallback?.call(this, v);
                        this.clampPoints();
                        return result;
                    };
                }
                
                return r;
            };
            
            // Add method to clamp points to min/max range
            nodeType.prototype.clampPoints = function() {
                if (!this.points || this.points.length === 0) return;
                
                // Get current min/max values
                const minValue = this.widgets.find(w => w.name === "min_value")?.value ?? 0;
                const maxValue = this.widgets.find(w => w.name === "max_value")?.value ?? 1;
                
                // Clamp any out-of-bounds points
                let needsUpdate = false;
                this.points = this.points.map(([frame, value]) => {
                    const clampedValue = Math.min(Math.max(value, minValue), maxValue);
                    if (clampedValue !== value) needsUpdate = true;
                    return [frame, clampedValue];
                });
                
                // Only update if points were actually clamped
                if (needsUpdate) {
                    this.updatePointsValue();
                    this.setDirtyCanvas(true, true);
                }
            };
            
            // Calculate widget area height
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
            
            // Get current frame count value, checking both widget and input
            nodeType.prototype.getCurrentFrameCount = function() {
                // First check if we have a connected input
                const frameCountInput = this.inputs.find(input => input.name === "frame_count");
                if (frameCountInput && frameCountInput.link !== null) {
                    const link = this.graph.links[frameCountInput.link];
                    if (link) {
                        const inputNode = this.graph._nodes_by_id[link.origin_id];
                        if (inputNode) {
                            // Try to get value from node's widget first
                            const widget = inputNode.widgets?.[0];
                            if (widget) {
                                // console.log("Getting frame count from widget:", widget.value);
                                return widget.value;
                            }
                            // Fallback to output value if no widget
                            const frameCount = inputNode.getOutputData(link.origin_slot);
                            // console.log("Getting frame count from output:", frameCount);
                            return frameCount;
                        }
                    }
                }
                
                // Fallback to widget value
                const frameCountWidget = this.widgets.find(w => w.name === "frame_count");
                return frameCountWidget?.value || 30;  // Default to 30 if no value found
            };

            // Override onDrawForeground to draw the graph
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                
                if (this.flags.collapsed) return;
                
                // Calculate graph dimensions based on node size and widget area
                const margin = 30;
                const widgetAreaHeight = this.getWidgetAreaHeight();
                const graphWidth = this.size[0] - 2 * margin;
                const graphHeight = Math.max(200, this.size[1] - widgetAreaHeight - margin * 2);
                const graphY = widgetAreaHeight + margin;
                
                // Draw background
                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(margin, graphY, graphWidth, graphHeight);
                ctx.strokeStyle = "#666";
                ctx.strokeRect(margin, graphY, graphWidth, graphHeight);
                
                // Draw help text
                ctx.fillStyle = "#888";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Click empty space to add • Double-click point to delete • Drag points to move", 
                    margin + graphWidth / 2, graphY + 20);
                
                // Draw grid
                ctx.strokeStyle = "#333";
                ctx.lineWidth = 0.5;
                
                // Vertical lines (frames) and labels
                const frameCount = this.getCurrentFrameCount();
                const maxFrame = frameCount - 1;  // Maximum valid frame
                const frameStep = Math.max(1, Math.floor(maxFrame / 10));
                ctx.fillStyle = "#888";
                ctx.font = "10px Arial";
                ctx.textAlign = "center";
                
                for (let f = 0; f <= maxFrame; f += frameStep) {
                    const x = margin + (f / maxFrame) * graphWidth;
                    ctx.beginPath();
                    ctx.moveTo(x, graphY);
                    ctx.lineTo(x, graphY + graphHeight);
                    ctx.stroke();
                    // Frame number labels
                    ctx.fillText(f.toString(), x, graphY + graphHeight + 15);
                }
                
                // Get min/max values from widgets
                const minValue = this.widgets.find(w => w.name === "min_value")?.value ?? 0;
                const maxValue = this.widgets.find(w => w.name === "max_value")?.value ?? 1;
                const valueRange = maxValue - minValue;
                
                // Horizontal lines (values) and labels
                ctx.textAlign = "right";
                for (let v = 0; v <= 1; v += 0.1) {
                    const y = graphY + v * graphHeight;
                    ctx.beginPath();
                    ctx.moveTo(margin, y);
                    ctx.lineTo(margin + graphWidth, y);
                    ctx.stroke();
                    // Value labels
                    const value = maxValue - (v * valueRange);
                    ctx.fillText(value.toFixed(1), margin - 5, y + 4);
                }
                
                // Axis labels
                ctx.save();
                ctx.translate(margin - 25, graphY + graphHeight / 2);
                ctx.rotate(-Math.PI / 2);
                ctx.textAlign = "center";
                ctx.fillText("Value", 0, 0);
                ctx.restore();
                
                ctx.textAlign = "center";
                ctx.fillText("Frame", margin + graphWidth / 2, graphY + graphHeight + 30);
                
                // Draw points and lines
                if (this.points && this.points.length > 0) {
                    // Draw lines between points
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    const points = this.points.map(([frame, value]) => ({
                        x: margin + (frame / maxFrame) * graphWidth,
                        y: graphY + (1 - this.normalizeValue(value)) * graphHeight
                    }));
                    
                    ctx.moveTo(points[0].x, points[0].y);
                    for (let i = 1; i < points.length; i++) {
                        ctx.lineTo(points[i].x, points[i].y);
                    }
                    ctx.stroke();
                    
                    // Draw points with hover and selection effects
                    points.forEach((point, i) => {
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, i === this.selectedPoint ? 7 : 5, 0, Math.PI * 2);
                        
                        if (i === this.selectedPoint) {
                            ctx.fillStyle = "#00ff00";
                        } else if (i === this.hoverPoint) {
                            ctx.fillStyle = "#ffff00";
                        } else {
                            ctx.fillStyle = "#fff";
                        }
                        
                        ctx.fill();
                        
                        if (i === this.selectedPoint || i === this.hoverPoint) {
                            ctx.strokeStyle = "#000";
                            ctx.lineWidth = 2;
                            ctx.stroke();
                        }
                    });
                }
                
                // Draw potential new point position
                if (this.isAddingPoint && this.mousePos) {
                    const [frame, value] = this.coordsToGraphValues(this.mousePos[0], this.mousePos[1]);
                    if (frame >= 0 && frame < frameCount) {
                        const x = margin + (frame / maxFrame) * graphWidth;
                        const y = graphY + (1 - this.normalizeValue(value)) * graphHeight;
                        
                        ctx.beginPath();
                        ctx.arc(x, y, 5, 0, Math.PI * 2);
                        ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
                        ctx.fill();
                        ctx.strokeStyle = "#fff";
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                }
            };
            
            // Normalize value to 0-1 range based on min/max widgets
            nodeType.prototype.normalizeValue = function(value) {
                const minValue = this.widgets.find(w => w.name === "min_value")?.value ?? 0;
                const maxValue = this.widgets.find(w => w.name === "max_value")?.value ?? 1;
                return (value - minValue) / (maxValue - minValue);
            };
            
            // Denormalize value from 0-1 range to min/max range
            nodeType.prototype.denormalizeValue = function(normalized) {
                const minValue = this.widgets.find(w => w.name === "min_value")?.value ?? 0;
                const maxValue = this.widgets.find(w => w.name === "max_value")?.value ?? 1;
                return normalized * (maxValue - minValue) + minValue;
            };
            
            // Convert coordinates to graph values
            nodeType.prototype.coordsToGraphValues = function(x, y) {
                const margin = 30;
                const widgetAreaHeight = this.getWidgetAreaHeight();
                const graphWidth = this.size[0] - 2 * margin;
                const graphHeight = Math.max(200, this.size[1] - widgetAreaHeight - margin * 2);
                const graphY = widgetAreaHeight + margin;
                
                const frameCount = this.getCurrentFrameCount();
                const maxFrame = frameCount - 1;
                
                // Calculate frame (x value) and clamp to valid range
                const frame = Math.round(((x - margin) / graphWidth) * maxFrame);
                const clampedFrame = Math.max(0, Math.min(maxFrame, frame));
                
                // Calculate value (y value) - normalize based on graph position
                const normalizedY = Math.max(0, Math.min(1, (y - graphY) / graphHeight));
                const value = this.denormalizeValue(1 - normalizedY);
                
                return [clampedFrame, value];
            };
            
            // Check if mouse is over graph area
            nodeType.prototype.isMouseOverGraph = function(x, y) {
                const margin = 30;
                const widgetAreaHeight = this.getWidgetAreaHeight();
                const graphWidth = this.size[0] - 2 * margin;
                const graphHeight = Math.max(200, this.size[1] - widgetAreaHeight - margin * 2);
                const graphY = widgetAreaHeight + margin;
                
                return x >= margin && 
                       x <= this.size[0] - margin && 
                       y >= graphY && 
                       y <= graphY + graphHeight;
            };
            
            // Find point near coordinates
            nodeType.prototype.findNearPoint = function(x, y) {
                const margin = 30;
                const widgetAreaHeight = this.getWidgetAreaHeight();
                const graphWidth = this.size[0] - 2 * margin;
                const graphHeight = Math.max(200, this.size[1] - widgetAreaHeight - margin * 2);
                const graphY = widgetAreaHeight + margin;
                const frameCount = this.widgets.find(w => w.name === "frame_count")?.value || 30;
                const maxFrame = frameCount - 1;
                
                for (let i = 0; i < this.points.length; i++) {
                    const [frame, value] = this.points[i];
                    const px = margin + (frame / maxFrame) * graphWidth;
                    const py = graphY + (1 - this.normalizeValue(value)) * graphHeight;
                    const dist = Math.sqrt((x - px) ** 2 + (y - py) ** 2);
                    if (dist < 10) return i;
                }
                return null;
            };
            
            // Update the hidden points value
            nodeType.prototype.updatePointsValue = function() {
                const pointsStr = JSON.stringify(this.points);
                // Find or create a hidden widget to store the points
                let pointsWidget = this.widgets.find(w => w.name === "points");
                if (!pointsWidget) {
                    // Create new points widget at the end of the list
                    pointsWidget = {
                        type: "text",
                        name: "points",
                        value: "[]",
                        hidden: true
                    };
                    this.widgets.push(pointsWidget);
                }
                pointsWidget.value = pointsStr;
                // Trigger widget change to ensure value is saved
                if (this.onWidgetChanged) {
                    this.onWidgetChanged(pointsWidget.name, pointsWidget.value, pointsWidget.value, pointsWidget);
                }
            };
            
            // Override mouse handlers for improved drawing
            nodeType.prototype.onMouseDown = function(e, pos) {
                const [x, y] = pos;
                if (!this.isMouseOverGraph(x, y)) return false;
                
                const pointIndex = this.findNearPoint(x, y);
                
                if (pointIndex !== null) {
                    if (this.isDragging && this.selectedPoint !== null) {
                        // If already dragging and clicked, drop the point
                        this.isDragging = false;
                        this.selectedPoint = null;
                        this.dragStartPos = null;
                    } else {
                        // Start dragging
                        this.selectedPoint = pointIndex;
                        this.dragStartPos = [...pos];
                        this.isDragging = true;
                    }
                } else {
                    // Not on a point - add new point
                    const [frame, value] = this.coordsToGraphValues(x, y);
                    const frameCount = this.widgets.find(w => w.name === "frame_count")?.value || 30;
                    const maxFrame = frameCount - 1;
                    if (frame >= 0 && frame <= maxFrame) {
                        // Remove any existing point at the same frame
                        const existingIndex = this.points.findIndex(p => p[0] === frame);
                        if (existingIndex !== -1) {
                            this.points.splice(existingIndex, 1);
                        }
                        
                        this.points.push([frame, value]);
                        this.points.sort((a, b) => a[0] - b[0]);
                        this.selectedPoint = this.points.findIndex(p => p[0] === frame);
                        this.updatePointsValue();
                    }
                }
                
                this.setDirtyCanvas(true, true);
                return true;
            };
            
            nodeType.prototype.onMouseMove = function(e, pos) {
                const [x, y] = pos;
                this.mousePos = pos;
                
                if (!this.isMouseOverGraph(x, y)) {
                    // Clear all states when mouse leaves graph area
                    this.hoverPoint = null;
                    this.isAddingPoint = false;
                    this.selectedPoint = null;
                    this.isDragging = false;
                    this.dragStartPos = null;
                    this.setDirtyCanvas(true, true);
                    return false;
                }
                
                if (this.isDragging && this.selectedPoint !== null) {
                    const [frame, value] = this.coordsToGraphValues(x, y);
                    const frameCount = this.widgets.find(w => w.name === "frame_count")?.value || 30;
                    const maxFrame = frameCount - 1;
                    // Keep strict < frameCount for dragging to prevent edge issues
                    if (frame >= 0 && frame <= maxFrame) {
                        // Check if there's already a point at the target frame (except selected point)
                        const existingIndex = this.points.findIndex((p, i) => 
                            i !== this.selectedPoint && p[0] === frame);
                        
                        if (existingIndex === -1) {
                            this.points[this.selectedPoint] = [frame, value];
                            this.points.sort((a, b) => a[0] - b[0]);
                            this.selectedPoint = this.points.findIndex(p => 
                                p[0] === frame && p[1] === value);
                            this.updatePointsValue();
                        }
                    }
                } else {
                    // Update hover state
                    this.hoverPoint = this.findNearPoint(x, y);
                    this.isAddingPoint = this.hoverPoint === null;
                }
                
                this.setDirtyCanvas(true, true);
                return true;
            };
            
            nodeType.prototype.onMouseUp = function(e, pos) {
                // Only clear states if we're not in dragging mode
                if (!this.isDragging) {
                    this.selectedPoint = null;
                    this.dragStartPos = null;
                }
                this.mouseDownTime = null;
                this.setDirtyCanvas(true, true);
                return false;
            };

            // Add double click handler
            nodeType.prototype.onDblClick = function(e, pos) {
                const [x, y] = pos;
                if (!this.isMouseOverGraph(x, y)) return false;
                
                const pointIndex = this.findNearPoint(x, y);
                if (pointIndex !== null) {
                    this.points.splice(pointIndex, 1);
                    this.updatePointsValue();
                    // Clear all interaction states
                    this.selectedPoint = null;
                    this.isDragging = false;
                    this.dragStartPos = null;
                    this.setDirtyCanvas(true, true);
                    return true;
                }
                return false;
            };

            // Handle node resizing
            nodeType.prototype.onResize = function(size) {
                if (onResize) {
                    onResize.apply(this, arguments);
                }
                // Ensure minimum size
                this.size[0] = Math.max(400, size[0]);
                this.size[1] = Math.max(500, size[1]);
                // Force graph area recalculation
                this.setDirtyCanvas(true, true);
            };

            // Add method to update node size
            nodeType.prototype.setSize = function(size) {
                this.size = size;
                if (this.onResize) {
                    this.onResize(size);
                }
            };

            // Add click handler
            nodeType.prototype.onClick = function(e, pos) {
                // Clear all states on any click as a safety measure
                this.selectedPoint = null;
                this.isDragging = false;
                this.dragStartPos = null;
                this.setDirtyCanvas(true, true);
                return false;
            };

            // Template loading method
            nodeType.prototype.loadTemplate = function(templateName) {
                const frameCount = this.getCurrentFrameCount();
                const minValue = this.widgets.find(w => w.name === "min_value")?.value ?? 0;
                const maxValue = this.widgets.find(w => w.name === "max_value")?.value ?? 1;
                const cyclesWidget = this.widgets.find(w => w.name === "template_cycles");
                const cycles = cyclesWidget?.value ?? 1;
                
                if (templateName in FeatureTemplates) {
                    this.points = FeatureTemplates[templateName](frameCount, minValue, maxValue, cycles);
                    this.updatePointsValue();
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    }
}); 