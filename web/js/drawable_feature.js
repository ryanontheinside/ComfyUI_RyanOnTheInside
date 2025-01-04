import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Register custom widget type
app.registerExtension({
    name: "RyanOnTheInside.DrawableFeature",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("Registering DrawableFeature extension", { nodeType, nodeData });
        
        if (nodeData.name === "DrawableFeatureNode") {
            console.log("Found DrawableFeatureNode, setting up widget");
            
            // Set default size
            nodeType.size = [400, 500];  // Increased default height
            
            // Store the original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            const onMouseDown = nodeType.prototype.onMouseDown;
            const onMouseMove = nodeType.prototype.onMouseMove;
            const onMouseUp = nodeType.prototype.onMouseUp;
            const onDblClick = nodeType.prototype.onDblClick;
            const onResize = nodeType.prototype.onResize;
            
            // Override onNodeCreated to initialize the node
            nodeType.prototype.onNodeCreated = function() {
                console.log("Node created, initializing node");
                const r = onNodeCreated?.apply(this, arguments);
                
                // Remove default points widget
                const pointsWidget = this.widgets.find(w => w.name === "points");
                if (pointsWidget) {
                    const index = this.widgets.indexOf(pointsWidget);
                    if (index > -1) {
                        this.widgets.splice(index, 1);
                    }
                }
                
                // Initialize points array and state
                this.points = [];
                this.isDragging = false;
                this.selectedPoint = null;
                this.isDrawing = false;
                this.lastDrawnFrame = null;  // Track last drawn frame to avoid duplicates
                
                // Add clear button widget
                this.addWidget("button", "Clear Graph", "clear", () => {
                    this.points = [];
                    this.updatePointsValue();
                    this.setDirtyCanvas(true, true);
                });
                
                return r;
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
                
                // Draw mode indicator
                if (this.isDrawing) {
                    ctx.fillStyle = "rgba(0, 255, 0, 0.2)";
                    ctx.fillRect(margin, graphY, graphWidth, graphHeight);
                    ctx.fillStyle = "#888";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Draw Mode: Click to add points", margin + graphWidth / 2, graphY + 20);
                }
                
                // Draw grid
                ctx.strokeStyle = "#333";
                ctx.lineWidth = 0.5;
                
                // Vertical lines (frames) and labels
                const frameCount = this.widgets.find(w => w.name === "frame_count")?.value || 30;
                const frameStep = Math.max(1, Math.floor(frameCount / 10));
                ctx.fillStyle = "#888";
                ctx.font = "10px Arial";
                ctx.textAlign = "center";
                
                for (let f = 0; f <= frameCount; f += frameStep) {
                    const x = margin + (f / frameCount) * graphWidth;
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
                    
                    const [firstFrame, firstValue] = this.points[0];
                    const firstX = margin + (firstFrame / frameCount) * graphWidth;
                    const firstY = graphY + (1 - this.normalizeValue(firstValue)) * graphHeight;
                    ctx.moveTo(firstX, firstY);
                    
                    for (let i = 1; i < this.points.length; i++) {
                        const [frame, value] = this.points[i];
                        const x = margin + (frame / frameCount) * graphWidth;
                        const y = graphY + (1 - this.normalizeValue(value)) * graphHeight;
                        ctx.lineTo(x, y);
                    }
                    ctx.stroke();
                    
                    // Draw points
                    ctx.fillStyle = "#fff";
                    for (const [frame, value] of this.points) {
                        const x = margin + (frame / frameCount) * graphWidth;
                        const y = graphY + (1 - this.normalizeValue(value)) * graphHeight;
                        ctx.beginPath();
                        ctx.arc(x, y, 5, 0, Math.PI * 2);
                        ctx.fill();
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
            
            // Check if mouse is over graph area
            nodeType.prototype.isMouseOverGraph = function(x, y) {
                const margin = 30;
                const graphHeight = Math.max(200, this.size[1] - 200);
                const graphY = this.size[1] - graphHeight - margin;
                return x >= margin && x <= this.size[0] - margin && y >= graphY && y <= graphY + graphHeight;
            };
            
            // Convert coordinates to graph values
            nodeType.prototype.coordsToGraphValues = function(x, y) {
                const margin = 30;
                const graphWidth = this.size[0] - 2 * margin;
                const graphHeight = Math.max(200, this.size[1] - 200);
                const graphY = this.size[1] - graphHeight - margin;
                
                const frameCount = this.widgets.find(w => w.name === "frame_count")?.value || 30;
                const frame = Math.round(((x - margin) / graphWidth) * frameCount);
                const normalizedValue = Math.max(0, Math.min(1, 1 - (y - graphY) / graphHeight));
                const value = this.denormalizeValue(normalizedValue);
                
                return [frame, value];
            };
            
            // Find point near coordinates
            nodeType.prototype.findNearPoint = function(x, y) {
                const margin = 30;
                const graphWidth = this.size[0] - 2 * margin;
                const graphHeight = Math.max(200, this.size[1] - 200);
                const graphY = this.size[1] - graphHeight - margin;
                const frameCount = this.widgets.find(w => w.name === "frame_count")?.value || 30;
                
                for (let i = 0; i < this.points.length; i++) {
                    const [frame, value] = this.points[i];
                    const px = margin + (frame / frameCount) * graphWidth;
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
                    pointsWidget = this.addWidget("text", "points", "[]", null);
                    pointsWidget.hidden = true;
                }
                pointsWidget.value = pointsStr;
            };
            
            // Override mouse handlers for improved drawing
            nodeType.prototype.onMouseDown = function(e, pos, ctx) {
                if (onMouseDown) {
                    const r = onMouseDown.apply(this, arguments);
                    if (r) return r;
                }
                
                const [x, y] = pos;
                if (this.isMouseOverGraph(x, y)) {
                    // Check for Ctrl+click to delete
                    if (e.ctrlKey || e.metaKey) {  // metaKey for Mac
                        const pointIndex = this.findNearPoint(x, y);
                        if (pointIndex !== null) {
                            this.points.splice(pointIndex, 1);
                            this.updatePointsValue();
                            this.setDirtyCanvas(true, true);
                            return true;
                        }
                    } else {
                        const pointIndex = this.findNearPoint(x, y);
                        if (pointIndex !== null) {
                            // Start dragging existing point
                            this.isDragging = true;
                            this.selectedPoint = pointIndex;
                        } else {
                            // Start drawing mode
                            this.isDrawing = true;
                            const [frame, value] = this.coordsToGraphValues(x, y);
                            if (frame >= 0 && frame < (this.widgets.find(w => w.name === "frame_count")?.value || 30)) {
                                this.points.push([frame, value]);
                                this.lastDrawnFrame = frame;
                                this.points.sort((a, b) => a[0] - b[0]);
                                this.updatePointsValue();
                                this.setDirtyCanvas(true, true);
                            }
                        }
                    }
                    return true;
                }
                
                return false;
            };
            
            nodeType.prototype.onMouseMove = function(e, pos) {
                if (onMouseMove) {
                    const r = onMouseMove.apply(this, arguments);
                    if (r) return r;
                }
                
                if (this.isDragging && this.selectedPoint !== null) {
                    // Handle dragging existing point
                    const [x, y] = pos;
                    const [frame, value] = this.coordsToGraphValues(x, y);
                    if (frame >= 0 && frame < (this.widgets.find(w => w.name === "frame_count")?.value || 30)) {
                        this.points[this.selectedPoint] = [frame, value];
                        this.points.sort((a, b) => a[0] - b[0]);
                        this.selectedPoint = this.points.findIndex(p => p[0] === frame);
                        this.updatePointsValue();
                        this.setDirtyCanvas(true, true);
                        return true;
                    }
                } else if (this.isDrawing) {
                    // Handle continuous drawing
                    const [x, y] = pos;
                    if (this.isMouseOverGraph(x, y)) {
                        const [frame, value] = this.coordsToGraphValues(x, y);
                        if (frame >= 0 && frame < (this.widgets.find(w => w.name === "frame_count")?.value || 30)) {
                            // Only add point if we're on a new frame
                            if (frame !== this.lastDrawnFrame) {
                                this.points.push([frame, value]);
                                this.lastDrawnFrame = frame;
                                this.points.sort((a, b) => a[0] - b[0]);
                                this.updatePointsValue();
                                this.setDirtyCanvas(true, true);
                            }
                        }
                    }
                    return true;
                }
                
                return false;
            };
            
            nodeType.prototype.onMouseUp = function(e, pos) {
                if (onMouseUp) {
                    const r = onMouseUp.apply(this, arguments);
                    if (r) return r;
                }
                
                this.isDragging = false;
                this.selectedPoint = null;
                this.isDrawing = false;
                this.lastDrawnFrame = null;
                return false;
            };

            // Remove double-click handler since we're using Ctrl+click for deletion
            nodeType.prototype.onDblClick = null;

            // Handle node resizing
            nodeType.prototype.onResize = function(size) {
                if (onResize) {
                    onResize.apply(this, arguments);
                }
                // Ensure minimum size
                this.size[0] = Math.max(400, size[0]);
                this.size[1] = Math.max(500, size[1]);
                this.setDirtyCanvas(true, true);
            };
        }
    }
}); 