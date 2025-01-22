// Extension to add ctrl-click functionality for widget/input conversion
(function() {
    // Core ComfyUI functions for widget conversion
    function showWidget(widget) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;
        widget.serializeValue = widget.origSerializeValue;
        delete widget.origType;
        delete widget.origComputeSize;
        delete widget.origSerializeValue;
        if (widget.linkedWidgets) {
            for (const w of widget.linkedWidgets) {
                showWidget(w);
            }
        }
    }

    function convertToWidget(node, widget) {
        showWidget(widget);
        const [oldWidth, oldHeight] = node.size;
        
        // Find and remove the input
        const inputIndex = node.inputs.findIndex((i) => i.widget?.name === widget.name);
        if (inputIndex !== -1) {
            node.removeInput(inputIndex);
        }

        // Make sure the widget is in the node's widgets array
        if (!node.widgets.includes(widget)) {
            node.widgets.push(widget);
        }

        // Adjust widget positions
        for (const widget2 of node.widgets) {
            widget2.last_y -= LiteGraph.NODE_SLOT_HEIGHT;
        }

        node.setSize([
            Math.max(oldWidth, node.size[0]),
            Math.max(oldHeight, node.size[1])
        ]);
    }

    class WidgetHotkeyHandler {
        constructor(app) {
            this.app = app;
            // console.log("[Widget Hotkey] Starting to load extension...");
            this.setupEventHandler();
        }

        setupEventHandler() {
            if (!this.app.canvas) {
                // console.log("[Widget Hotkey] Canvas not available yet, waiting...");
                setTimeout(() => this.setupEventHandler(), 100);
                return;
            }

            // console.log("[Widget Hotkey] Setting up event handler...");
            
            // Store the original mousedown handler
            const originalMouseDown = this.app.canvas.onMouseDown;
            
            // Override the mousedown handler
            this.app.canvas.onMouseDown = (e) => {
                if (e.ctrlKey) {
                    // Get the node under the mouse
                    const node = this.app.graph.getNodeOnPos(e.canvasX, e.canvasY);
                    if (node) {
                        // Convert canvas position to node local position
                        const localPos = [
                            e.canvasX - node.pos[0],
                            e.canvasY - node.pos[1]
                        ];

                        // Find widget at position
                        const widget = this.findWidgetAtPos(node, localPos);
                        
                        if (widget) {
                            return this.handleWidgetClick(node, widget);
                        }
                    }
                }

                // Call the original handler
                if (originalMouseDown) {
                    return originalMouseDown.call(this.app.canvas, e);
                }
            };

            // console.log("[Widget Hotkey] Event handler setup complete!");
        }

        findWidgetAtPos(node, localPos) {
            // First check input slots (converted widgets)
            let y = LiteGraph.NODE_TITLE_HEIGHT;
            for (const input of node.inputs || []) {
                if (input.widget) {
                    // Check if click is in input slot area
                    if (localPos[0] >= 0 && 
                        localPos[0] <= 20 && 
                        localPos[1] >= y && 
                        localPos[1] <= y + LiteGraph.NODE_SLOT_HEIGHT) {
                        // Return the widget from the input
                        return input.widget;
                    }
                }
                y += LiteGraph.NODE_SLOT_HEIGHT;
            }

            // Then check regular widgets using the same logic as tooltips
            for (const w of node.widgets || []) {
                // Skip widgets that are already converted to inputs
                if (w.type?.startsWith("converted-widget")) continue;

                let widgetWidth, widgetHeight;
                if (w.computeSize) {
                    [widgetWidth, widgetHeight] = w.computeSize(node.size[0]);
                } else {
                    widgetWidth = w.width || node.size[0];
                    widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT;
                }

                if (w.last_y !== undefined && 
                    localPos[0] >= 6 && 
                    localPos[0] <= widgetWidth - 12 && 
                    localPos[1] >= w.last_y && 
                    localPos[1] <= w.last_y + widgetHeight) {
                    return w;
                }
            }

            return null;
        }

        handleWidgetClick(node, widget) {
            // Check if this widget is from an input (it's already converted)
            const isInput = node.inputs.some(input => input.widget === widget);
            
            if (isInput) {
                // TODO
                // Convert input back to widget using core ComfyUI's convertToWidget function
                // console.log("[Widget Hotkey] Attempting to convert input back to widget:", widget.name);
                // convertToWidget(node, widget);
            } else {
                // Convert widget to input using node's built-in method
                // console.log("[Widget Hotkey] Attempting to convert widget to input:", widget.name);
                node.convertWidgetToInput(widget);
            }
            
            return false;
        }
    }

    function registerWidgetHotkeyExtension() {
        const app = window.app || window.ComfyApp || window.comfyAPI?.app?.app;
        if (app && app.registerExtension) {
            app.registerExtension({
                name: "RyanOnTheInside.WidgetInputHotkey",
                setup(app) {
                    // console.log("[Widget Hotkey] Setup phase starting...");
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