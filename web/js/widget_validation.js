// Widget validation for RyanOnTheInside Scheduler nodes
import { app } from "../../../scripts/app.js";

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
                    this.targetConstraints = {
                        min: options.min,
                        max: options.max,
                        step: options.step,
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

            return result;
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
                const targetMin = this.targetConstraints?.min ?? lowerWidget.options.min;
                const targetMax = this.targetConstraints?.max ?? upperWidget.options.max;
                const step = this.targetConstraints?.step ?? 1;

                // Function to snap value to nearest step
                const snapToStep = (value) => {
                    return Math.round(value / step) * step;
                };

                // Ensure lower threshold is less than upper threshold and both are within target constraints
                let lowerValue = Math.max(targetMin, Math.min(lowerWidget.value, upperWidget.value));
                let upperValue = Math.max(lowerWidget.value, Math.min(upperWidget.value, targetMax));

                // Snap values to steps
                lowerValue = snapToStep(lowerValue);
                upperValue = snapToStep(upperValue);

                // Round if integer type
                if (this.targetConstraints?.isInt) {
                    lowerValue = Math.round(lowerValue);
                    upperValue = Math.round(upperValue);
                }

                lowerWidget.value = lowerValue;
                upperWidget.value = upperValue;
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