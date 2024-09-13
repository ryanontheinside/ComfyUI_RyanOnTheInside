import { app } from "../../scripts/app.js";
import { drawPiano, handlePianoMouseDown } from "./piano.js";

console.log("Pitch Extension: FILE OPEN PitchRangeByNoteNode");
app.registerExtension({
    name: "org.ryanontheinside.pitch_range_by_note",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("nodeData.name:", nodeData.name);
        if (nodeData.name === "PitchRangeByNoteNode") {
            console.log("Pitch Extension: Modifying PitchRangeByNoteNode");

            // Add the disableWidget method to the nodeType prototype
            nodeType.prototype.disableWidget = function(widget) {
                if (widget) {
                    widget.disabled = true;
                    widget.visibleWidth = 0;
                    widget.onMouseDown = () => {};
                    widget.onMouseMove = () => {};
                    widget.onMouseUp = () => {};
                }
            };

            nodeType.prototype.onNodeCreated = function() {
                console.log("Pitch Extension: onNodeCreated called for PitchRangeByNoteNode");
                this.properties = this.properties || {};

                // Initialize selectedNotes
                this.properties.notes = this.properties.notes || "";

                // Initialize widgets
                this.initializeWidgets();

                // Load any saved state
                this.loadState();
            };

            nodeType.prototype.initializeWidgets = function() {
                // Disable the 'notes' widget and use it to display selected notes
                const notesWidget = this.widgets.find(w => w.name === "notes");
                if (notesWidget) {
                    this.disableWidget(notesWidget);
                } else {
                    this.addWidget("text", "notes", "", null, { name: "notes" });
                    this.disableWidget(this.widgets.find(w => w.name === "notes"));
                }
            };

            nodeType.prototype.syncNotesWidget = function() {
                const notesWidget = this.widgets.find(widget => widget.name === "notes");
                if (notesWidget) {
                    const selectedNotes = this.properties.notes.split(',')
                        .map(note => note.trim())
                        .filter(note => note !== '');
                    const selectedNoteNames = selectedNotes
                        .sort((a, b) => parseInt(a, 10) - parseInt(b, 10))
                        .map(midiNote => this.midiToNoteName(parseInt(midiNote, 10)))
                        .join(", ");
                    if (notesWidget.value !== selectedNoteNames) {
                        notesWidget.value = selectedNoteNames;
                    }
                }
                // Update internal state and save
                this.onPropertyChanged('notes');
                if (typeof this.saveState === 'function') {
                    this.saveState();
                }
            };

            nodeType.prototype.midiToNoteName = function(midiNote) {
                const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
                const octave = Math.floor(midiNote / 12) - 1;
                const noteIndex = midiNote % 12;
                return `${noteNames[noteIndex]}${octave}`;
            };

            // Save the original onDrawForeground method
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

            nodeType.prototype.saveState = function() {
                const state = {
                    notes: this.properties.notes
                };
                console.log("Saving state:", state);
                localStorage.setItem(`PitchRangeByNoteNode_${this.id}`, JSON.stringify(state));
            };

            nodeType.prototype.loadState = function() {
                const savedState = localStorage.getItem(`PitchRangeByNoteNode_${this.id}`);
                if (savedState) {
                    const state = JSON.parse(savedState);
                    console.log('Loaded state:', state);
                    if (state.notes) {
                        this.properties.notes = state.notes;
                    } else {
                        this.properties.notes = "";
                    }
                    this.syncNotesWidget();
                }
            };

            // Hook into the node added event to initialize the node
            const onAdded = app.graph.onNodeAdded;
            app.graph.onNodeAdded = function(node) {
                if (onAdded) {
                    onAdded.call(this, node);
                }
                if (node.type === "PitchRangeByNoteNode" && node.onNodeCreated) {
                    node.onNodeCreated();
                }
            };
        }
    },
});