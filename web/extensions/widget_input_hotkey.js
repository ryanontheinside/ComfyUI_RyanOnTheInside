// Extension to add shift-click functionality for widget/input conversion
(function() {
    class WidgetHotkeyHandler {
        constructor(app) {
            this.app = app;
            console.log("[Widget Hotkey] Starting to load extension...");
            this.setupEventHandler();
        }

        setupEventHandler() {
            if (!this.app.canvas) {
                console.log("[Widget Hotkey] Canvas not available yet, waiting...");
                setTimeout(() => this.setupEventHandler(), 100);
                return;
            }

            console.log("[Widget Hotkey] Setting up event handler...");
            
            // Store the original mousedown handler
            const originalMouseDown = this.app.canvas.onMouseDown;
            
            // Override the mousedown handler
            this.app.canvas.onMouseDown = (e) => {
                console.log("[Widget Hotkey] Canvas mouse down:", {
                    shift: e.ctrlKey,
                    canvasX: e.canvasX,
                    canvasY: e.canvasY,
                    api: !!this.app.api,
                    graph: !!this.app.graph
                });

                if (e.ctrlKey) {
                    // Get the node under the mouse
                    const node = this.app.graph.getNodeOnPos(e.canvasX, e.canvasY);
                    if (node) {
                        // Convert canvas position to node local position
                        const localPos = [
                            e.canvasX - node.pos[0],
                            e.canvasY - node.pos[1]
                        ];

                        // Debug all widget positions
                        if (node.widgets) {
                            console.log("[Widget Hotkey] All widgets:", node.widgets.map(w => ({
                                name: w.name,
                                type: w.type,
                                pos: w.last_pos,
                                computedPos: this.computeWidgetPos(node, w),
                                origType: w.origType,
                                isConverted: w.type?.startsWith("converted-widget")
                            })));
                        }

                        // Find widget at position
                        const widget = this.findWidgetAtPos(node, localPos);
                        
                        console.log("[Widget Hotkey] Detection results:", {
                            nodeTitle: node.title,
                            localPos,
                            foundWidget: widget ? widget.name : "none",
                            widgetType: widget ? widget.type : "none",
                            isConverted: widget ? widget.type?.startsWith("converted-widget") : false
                        });

                        if (widget) {
                            console.log("[Widget Hotkey] Widget found:", {
                                name: widget.name,
                                type: widget.type,
                                origType: widget.origType,
                                isConverted: widget.type?.startsWith("converted-widget")
                            });

                            return this.handleWidgetClick(node, widget);
                        }
                    }
                }

                // Call the original handler
                if (originalMouseDown) {
                    return originalMouseDown.call(this.app.canvas, e);
                }
            };

            console.log("[Widget Hotkey] Event handler setup complete!");
        }

        computeWidgetPos(node, widget) {
            const titleHeight = 30;
            const widgetHeight = 20;
            const widgetSpacing = 4;
            let y = titleHeight;

            // Find this widget's position
            for (const w of node.widgets) {
                // Skip converted widgets in position calculation
                if (w.type?.startsWith("converted-widget")) {
                    continue;
                }
                if (w === widget) {
                    return {
                        x: 10,
                        y: y,
                        width: node.size[0] - 20,
                        height: widgetHeight
                    };
                }
                y += widgetHeight + widgetSpacing;
            }

            // For converted widgets, check if they have a corresponding input
            if (widget.type?.startsWith("converted-widget")) {
                const inputIndex = node.inputs.findIndex(i => i.widget?.name === widget.name);
                if (inputIndex !== -1) {
                    return {
                        x: 0,
                        y: titleHeight + inputIndex * (widgetHeight + widgetSpacing),
                        width: 20,
                        height: widgetHeight
                    };
                }
            }
            return null;
        }

        findWidgetAtPos(node, localPos) {
            // Constants for layout
            const titleHeight = 30;
            const widgetHeight = 20;
            const widgetSpacing = 4;
            const slotHeight = LiteGraph.NODE_SLOT_HEIGHT;

            // Count number of inputs before each widget to adjust y position
            let inputCount = 0;
            let y = titleHeight;

            // First check regular widgets
            for (const widget of node.widgets) {
                // Skip converted widgets
                if (widget.type?.startsWith("converted-widget")) {
                    continue;
                }

                // Adjust y position based on inputs that come before this widget
                while (inputCount < node.inputs.length && 
                       (!node.inputs[inputCount].widget || 
                        node.inputs[inputCount].widget.name !== widget.name)) {
                    y += slotHeight;
                    inputCount++;
                }

                // Check if point is inside widget area
                if (localPos[0] >= 10 && 
                    localPos[0] <= node.size[0] - 10 && 
                    localPos[1] >= y && 
                    localPos[1] <= y + widgetHeight) {
                    return widget;
                }

                y += widgetHeight + widgetSpacing;
            }

            // Reset for checking converted widgets
            y = titleHeight;
            inputCount = 0;

            // Then check converted widgets (inputs)
            for (const input of node.inputs) {
                if (input.widget?.type?.startsWith("converted-widget")) {
                    // Check if point is inside input area
                    if (localPos[0] >= 0 && 
                        localPos[0] <= 20 && 
                        localPos[1] >= y && 
                        localPos[1] <= y + slotHeight) {
                        return input.widget;
                    }
                }
                y += slotHeight;
                inputCount++;
            }

            return null;
        }

        handleWidgetClick(node, widget) {
            // If it's already an input (has origType), convert back to widget
            if (widget.origType) {
                console.log("[Widget Hotkey] Converting input back to widget:", {
                    name: widget.name,
                    origType: widget.origType,
                    currentType: widget.type
                });
                this.convertToWidget(node, widget);
            } else {
                // Get widget config and check if convertible
                const config = [widget.type, widget.options || {}];
                console.log("[Widget Hotkey] Widget config:", {
                    widget: widget.name,
                    config: config,
                    type: widget.type,
                    options: widget.options
                });
                
                const isConvertible = this.isConvertibleWidget(widget, config);
                console.log("[Widget Hotkey] Is convertible:", {
                    widget: widget.name,
                    isConvertible: isConvertible,
                    type: widget.type,
                    configType: config[0],
                    forceInput: widget.options?.forceInput
                });
                
                if (isConvertible) {
                    console.log("[Widget Hotkey] Converting widget to input");
                    this.convertToInput(node, widget, config);
                    console.log("[Widget Hotkey] Conversion complete. New widget state:", {
                        name: widget.name,
                        type: widget.type,
                        origType: widget.origType
                    });
                } else {
                    console.log("[Widget Hotkey] Widget is not convertible");
                }
            }
            
            return false;
        }

        convertToWidget(node, widget) {
            console.log("[Widget Hotkey] Converting to widget:", widget.name);
            widget.type = widget.origType;
            widget.computeSize = widget.origComputeSize;
            widget.serializeValue = widget.origSerializeValue;
            delete widget.origType;
            delete widget.origComputeSize;
            delete widget.origSerializeValue;
            if (widget.linkedWidgets) {
                for (const w of widget.linkedWidgets) {
                    this.convertToWidget(node, w);
                }
            }
            
            const [oldWidth, oldHeight] = node.size;
            node.removeInput(node.inputs.findIndex((i) => i.widget?.name === widget.name));
            for (const widget2 of node.widgets) {
                widget2.last_y -= this.app.canvas.constructor.NODE_SLOT_HEIGHT;
            }
            node.setSize([
                Math.max(oldWidth, node.size[0]),
                Math.max(oldHeight, node.size[1])
            ]);
            
            // Add node redraw
            this.app.graph.setDirtyCanvas(true, true); // Mark both the node and connections as dirty
            this.app.graph.change(); // Trigger graph change event
        }

        recreateNode(node) {
            // Store the node's current state
            const id = node.id;
            const pos = [...node.pos];
            const size = [...node.size];
            const type = node.type;
            const inputs = [...node.inputs];
            const outputs = [...node.outputs];
            const properties = {...node.properties};
            const widgets = node.widgets ? node.widgets.map(w => ({...w})) : [];
            
            // Remove the old node
            this.app.graph.remove(node);
            
            // Create a new node of the same type
            const newNode = LiteGraph.createNode(type);
            newNode.id = id;
            newNode.pos = pos;
            newNode.size = size;
            
            // Configure the new node
            newNode.configure({
                id: id,
                type: type,
                pos: pos,
                size: size,
                properties: properties,
                inputs: inputs,
                outputs: outputs,
                widgets_values: widgets.map(w => w.value)
            });
            
            // Add it to the graph
            this.app.graph.add(newNode);
            
            return newNode;
        }

        convertToInput(node, widget, config) {
            const pos = node.widgets.indexOf(widget);
            if (pos === -1) return;

            widget.origType = widget.type;
            widget.origComputeSize = widget.computeSize;
            widget.origSerializeValue = widget.serializeValue;
            widget.type = "converted-widget";
            widget.computeSize = null;
            widget.serializeValue = null;

            const w = {
                name: widget.name,
                type: config[0],
                options: config[1],
                widget: widget,
            };

            node.addInput(node.inputs.length, w.type, w);
            
            // Recompute widget positions
            let y = 30; // Start after title
            for (const w of node.widgets) {
                if (!w.type?.startsWith("converted-widget")) {
                    w.last_y = y;
                    y += 20 + 4; // widget height + spacing
                }
            }

            // Recreate the node to ensure proper rendering
            this.recreateNode(node);
        }

        isConvertibleWidget(widget, config) {
            const VALID_TYPES = [
                "STRING",
                "combo",
                "number",
                "toggle",
                "BOOLEAN",
                "text",
                "string"
            ];
            return (VALID_TYPES.includes(widget.type) || VALID_TYPES.includes(config[0])) && !widget.options?.forceInput;
        }
    }

    function registerWidgetHotkeyExtension() {
        const app = window.app || window.ComfyApp || window.comfyAPI?.app?.app;
        if (app && app.registerExtension) {
            app.registerExtension({
                name: "RyanOnTheInside.WidgetInputHotkey",
                setup(app) {
                    console.log("[Widget Hotkey] Setup phase starting...");
                    // Create an instance of the handler class and initialize it
                    new WidgetHotkeyHandler(app);
                }
            });
        } else {
            // Try again after a short delay if app is not available yet
            setTimeout(registerWidgetHotkeyExtension, 100);
        }
    }

    // Start trying to register the extension
    registerWidgetHotkeyExtension();
})(); 