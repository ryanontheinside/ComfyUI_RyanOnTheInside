import { app } from "../../scripts/app.js";
import { drawPiano, handlePianoMouseDown } from "./piano.js";

app.registerExtension({
    name: "org.ryanontheinside.pitch_range_by_note",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PitchRangeByNoteNode") {

            nodeType.prototype.onNodeCreated = function() {
                this.properties = this.properties || {};
                this.properties.selectedNotes = []; // Initialize with an empty array
                this.pianoScroll = 0;
                this.scrollLeftIndicator = null;
                this.scrollRightIndicator = null;
                this.availableNotes = new Set(Array.from({length: 128}, (_, i) => i)); // All MIDI notes

                const notesWidget = this.widgets.find(widget => widget.name === "notes");
                if (notesWidget) {
                    notesWidget.hidden = true;
                    notesWidget.disabled = true; // Disable the widget immediately
                }

                // Increase the node's height and width
                const extraHeight = 80; // Piano height + margins
                const pianoWidth = 1020; // Total width of the piano
                this.size[1] += extraHeight;
                this.size[0] = Math.max(this.size[0], pianoWidth); // Set width to at least piano width plus margins

                // Ensure the node recomputes its size
                this.setDirtyCanvas(true, true);

                // Sync the notes widget initially
                this.syncNotesWidget();
            };

            nodeType.prototype.syncNotesWidget = function() {
                const notesWidget = this.widgets.find(widget => widget.name === "notes");
                if (notesWidget) {
                    const newValue = this.properties.selectedNotes.sort((a, b) => a - b).join(", ");
                    if (notesWidget.value !== newValue) {
                        notesWidget.value = newValue;
                        // Trigger a change event on the widget
                        if (notesWidget.callback) {
                            notesWidget.callback(newValue);
                        }
                    }
                }
                // Ensure the node updates its internal state
                this.onPropertyChanged('selectedNotes');
            };

            nodeType.prototype.onPropertyChanged = function(property) {
                if (property === 'selectedNotes') {
                    // Trigger any necessary updates or recomputation
                    this.setDirtyCanvas(true, true);
                }
            };

            // Save the original method
            nodeType.prototype._originalOnDrawForeground = nodeType.prototype.onDrawForeground;

            nodeType.prototype.onDrawForeground = function(ctx) {
                // Call the original onDrawForeground method if it exists
                if (typeof nodeType.prototype._originalOnDrawForeground === 'function') {
                    nodeType.prototype._originalOnDrawForeground.call(this, ctx);
                }

                if (this.flags.collapsed) return;

                // Draw widgets that are not hidden
                for (let w of this.widgets) {
                    if (w.hidden) {
                        continue;  // Skip drawing hidden widgets
                    }
                    if (w.draw) {
                        w.draw(ctx, this, this.pos);
                    }
                }

                drawPiano(ctx, this);
            };

            // Save the original onMouseDown method
            nodeType.prototype._originalOnMouseDown = nodeType.prototype.onMouseDown;

            nodeType.prototype.onMouseDown = function(event, pos, graphcanvas) {
                // Call the original onMouseDown method if it exists
                if (typeof nodeType.prototype._originalOnMouseDown === 'function') {
                    nodeType.prototype._originalOnMouseDown.call(this, event, pos, graphcanvas);
                }

                const [x, y] = pos;
                return handlePianoMouseDown(this, x, y);
            };

            // Disable the notes widget
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                onConnectionsChange?.apply(this, arguments);
                const notesWidget = this.widgets.find((w) => w.name === "notes");
                if (notesWidget) {
                    notesWidget.disabled = true;
                }
            };
        }
    },
});