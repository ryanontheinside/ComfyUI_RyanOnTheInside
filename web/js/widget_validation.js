// Widget validation for RyanOnTheInside Scheduler nodes
import { app } from "../../../scripts/app.js";

//TODO: add  validation for DrawableFeatureNode, frame count for instance. 
// Register validation behavior when nodes are connected
app.registerExtension({
    name: "RyanOnTheInside.WidgetValidation",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Check if this is one of our scheduler nodes
        if (!nodeData.category?.startsWith("RyanOnTheInside/FlexFeatures/Scheduling")) {
            return;
        }

        // Store constraints from connected widgets
        nodeType.prototype.targetConstraints = null;

        // Add handler for when node is created (including on page load)
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            // Function to check connections
            const checkConnections = () => {
                const outputLinks = this.outputs[0]?.links || [];
                
                for (const linkId of outputLinks) {
                    const link = app.graph.links[linkId];
                    if (!link) continue;

                    const targetNode = app.graph.getNodeById(link.target_id);
                    const targetSlot = link.target_slot;
                    
                    if (targetNode?.widgets) {
                        const inputName = targetNode.inputs[targetSlot]?.name;
                        const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                        
                        if (targetWidget?.options || (targetWidget?.type === "converted-widget" && targetWidget.options)) {
                            this.targetConstraints = {
                                min: targetWidget.options.min,
                                max: targetWidget.options.max,
                                step: targetWidget.options.step,
                                isInt: this.type === "FeatureToFlexIntParam"
                            };
                            
                            // Set initial values to match target widget's range
                            const lowerWidget = this.widgets.find(w => w.name === "lower_threshold");
                            const upperWidget = this.widgets.find(w => w.name === "upper_threshold");
                            if (lowerWidget && upperWidget) {
                                lowerWidget.value = this.targetConstraints.min;
                                upperWidget.value = this.targetConstraints.max;
                            }
                            
                            this.updateWidgetConstraints();
                        }
                    }
                }
            };

            // Check connections at different intervals to ensure graph is loaded
            checkConnections();
            setTimeout(checkConnections, 100);
            setTimeout(checkConnections, 1000);
            
            return result;
        };

        const originalOnConnectOutput = nodeType.prototype.onConnectOutput;
        nodeType.prototype.onConnectOutput = function (slot, type, input, targetNode, targetSlot) {
            const result = originalOnConnectOutput?.apply(this, arguments);

            if (targetNode?.widgets) {
                const inputName = targetNode.inputs[targetSlot]?.name;
                const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                
                if (targetWidget?.options || (targetWidget?.type === "converted-widget" && targetWidget.options)) {
                    const options = targetWidget.options;
                    console.log('Node type:', this.type);
                    console.log('Target widget options:', options);
                    
                    // Check if we already have constraints from another connection
                    if (this.targetConstraints) {
                        console.log('Warning: Already connected to another widget with constraints:', this.targetConstraints);
                        // Option 1: Prevent multiple connections
                        // return false; 
                        
                        // Option 2: Use the most restrictive constraints
                        this.targetConstraints = {
                            min: Math.max(parseFloat(options.min), this.targetConstraints.min),
                            max: Math.min(parseFloat(options.max), this.targetConstraints.max),
                            step: Math.max(parseFloat(options.step), this.targetConstraints.step),
                            isInt: this.type === "FeatureToFlexIntParam"
                        };
                        console.log('Updated to more restrictive constraints:', this.targetConstraints);
                    } else {
                        this.targetConstraints = {
                            min: parseFloat(options.min),
                            max: parseFloat(options.max),
                            step: parseFloat(options.step),
                            isInt: this.type === "FeatureToFlexIntParam"
                        };
                    }
                    
                    console.log('Target constraints:', this.targetConstraints);
                    
                    // Only set initial values if current values are outside constraints
                    const lowerWidget = this.widgets.find(w => w.name === "lower_threshold");
                    const upperWidget = this.widgets.find(w => w.name === "upper_threshold");
                    if (lowerWidget && upperWidget) {
                        const currentLower = parseFloat(lowerWidget.value);
                        const currentUpper = parseFloat(upperWidget.value);
                        
                        console.log('Current widget values:', {
                            lower: currentLower,
                            upper: currentUpper
                        });

                        // Only reset if values are outside constraints
                        if (currentLower < this.targetConstraints.min || 
                            currentLower > this.targetConstraints.max ||
                            currentUpper < this.targetConstraints.min || 
                            currentUpper > this.targetConstraints.max) {
                            console.log('Resetting values to match constraints');
                            lowerWidget.value = this.targetConstraints.min;
                            upperWidget.value = this.targetConstraints.max;
                        }
                    }
                    
                    this.updateWidgetConstraints();
                }
            }

            return result;
        };

        nodeType.prototype.onConnectionsChange = function(slotType, slot, isConnected, link_info, output) {
            // Handle disconnection
            if (!isConnected && slotType === LiteGraph.OUTPUT) {
                console.log('Output disconnected, checking remaining connections');
                
                // Reset constraints initially
                this.targetConstraints = null;
                
                // Check all remaining connections and rebuild constraints
                if (this.outputs[0].links && this.outputs[0].links.length > 0) {
                    console.log('Found remaining connections, rebuilding constraints');
                    
                    this.outputs[0].links.forEach(linkId => {
                        const link = this.graph.links[linkId];
                        if (!link) return;
                        
                        const targetNode = this.graph.getNodeById(link.target_id);
                        const targetSlot = link.target_slot;
                        
                        if (targetNode?.widgets) {
                            const inputName = targetNode.inputs[targetSlot]?.name;
                            const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                            
                            if (targetWidget?.options) {
                                const options = targetWidget.options;
                                console.log('Found target widget options:', options);
                                
                                // Update constraints based on remaining connection
                                if (this.targetConstraints) {
                                    // If we already have constraints, use the most restrictive ones
                                    this.targetConstraints = {
                                        min: Math.max(parseFloat(options.min), this.targetConstraints.min),
                                        max: Math.min(parseFloat(options.max), this.targetConstraints.max),
                                        step: Math.max(parseFloat(options.step), this.targetConstraints.step),
                                        isInt: this.type === "FeatureToFlexIntParam"
                                    };
                                } else {
                                    // First remaining connection sets initial constraints
                                    this.targetConstraints = {
                                        min: parseFloat(options.min),
                                        max: parseFloat(options.max),
                                        step: parseFloat(options.step),
                                        isInt: this.type === "FeatureToFlexIntParam"
                                    };
                                }
                                console.log('Updated constraints:', this.targetConstraints);
                            }
                        }
                    });
                } else {
                    console.log('No remaining connections, clearing constraints');
                    // Reset widgets to their default constraints if needed
                    const lowerWidget = this.widgets.find(w => w.name === "lower_threshold");
                    const upperWidget = this.widgets.find(w => w.name === "upper_threshold");
                    if (lowerWidget && upperWidget) {
                        // Reset to default widget options if they exist
                        if (lowerWidget.options) {
                            this.targetConstraints = {
                                min: parseFloat(lowerWidget.options.min),
                                max: parseFloat(upperWidget.options.max),
                                step: parseFloat(lowerWidget.options.step || 1),
                                isInt: this.type === "FeatureToFlexIntParam"
                            };
                        }
                    }
                }
                
                // Update widget constraints with new/cleared constraints
                this.updateWidgetConstraints();
            }
        };

        // Add method to update widget constraints
        nodeType.prototype.updateWidgetConstraints = function() {
            if (!this.targetConstraints) return;

            // Update thresholds based on target constraints
            const thresholdWidgets = ["lower_threshold", "upper_threshold"];
            thresholdWidgets.forEach(name => {
                const widget = this.widgets.find(w => w.name === name);
                if (widget) {
                    if (this.targetConstraints.min !== undefined) {
                        widget.options.min = this.targetConstraints.min;
                    }
                    if (this.targetConstraints.max !== undefined) {
                        widget.options.max = this.targetConstraints.max;
                    }
                    if (this.targetConstraints.step !== undefined) {
                        widget.options.step = this.targetConstraints.step;
                    }
                }
            });

            this.clampWidgetValues();
        };

        // Add method to clamp widget values
        nodeType.prototype.clampWidgetValues = function() {
            const lowerWidget = this.widgets.find(w => w.name === "lower_threshold");
            const upperWidget = this.widgets.find(w => w.name === "upper_threshold");

            if (lowerWidget && upperWidget) {
                const targetMin = parseFloat(this.targetConstraints?.min ?? lowerWidget.options.min);
                const targetMax = parseFloat(this.targetConstraints?.max ?? upperWidget.options.max);
                const step = parseFloat(this.targetConstraints?.step ?? 1);

                // Function to snap value to nearest step with decimal precision
                const snapToStep = (value) => {
                    // Special case: if value is the minimum or maximum, don't snap it
                    if (value === targetMin || value === targetMax) {
                        console.log(`Preserving exact value ${value} (min/max)`);
                        return value;
                    }
                    
                    // For other values, snap to nearest step but ensure we don't go below minimum
                    const snapped = Math.max(
                        targetMin,
                        Math.round(value / step) * step
                    );
                    console.log(`Snapping ${value} to step ${step}: ${snapped}`);
                    return snapped;
                };

                // Ensure lower threshold is less than upper threshold and both are within target constraints
                let lowerValue = Math.max(targetMin, Math.min(parseFloat(lowerWidget.value), parseFloat(upperWidget.value)));
                let upperValue = Math.max(parseFloat(lowerWidget.value), Math.min(parseFloat(upperWidget.value), targetMax));

                // Snap values to steps
                lowerValue = snapToStep(lowerValue);
                upperValue = snapToStep(upperValue);

                // Log values before integer rounding
                console.log('Before integer rounding:', {
                    lowerValue,
                    upperValue
                });

                // Round if integer type, otherwise maintain decimal precision
                if (this.targetConstraints?.isInt) {
                    lowerValue = Math.round(lowerValue);
                    upperValue = Math.round(upperValue);
                    console.log('After integer rounding:', {
                        lowerValue,
                        upperValue
                    });
                }

                lowerWidget.value = lowerValue;
                upperWidget.value = upperValue;

                // Log final values
                console.log('Final widget values:', {
                    lower: lowerWidget.value,
                    upper: upperWidget.value
                });
            }
        };

        // Override the widget's callback to enforce constraints
        const originalWidgetCallback = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function(widget, value) {
            if (["lower_threshold", "upper_threshold"].includes(widget.name)) {
                const step = this.targetConstraints?.step ?? 1;
                
                // Snap to step
                value = Math.round(value / step) * step;
                
                // Round if integer type
                if (this.targetConstraints?.isInt) {
                    value = Math.round(value);
                }
                
                widget.value = value;
                this.clampWidgetValues();
            }
            
            return originalWidgetCallback?.apply(this, [widget, value]);
        };
    }
}); 