import { app } from "../../scripts/app.js";
import { drawPiano, handlePianoMouseDown } from "./piano.js";

app.registerExtension({
    name: "org.ryanontheinside.midi_load_and_extract",
    
    // async setup() {
    //     console.log("MIDI Extension: Setup called");
    // },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MIDILoadAndExtract") {

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

            nodeType.prototype.uploadMIDI = function() {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = ".mid,.midi";
                input.onchange = (event) => {
                    const file = event.target.files[0];
                    if (file) {
                        const formData = new FormData();
                        formData.append("file", file);
            
                        fetch("/upload_midi", {
                            method: "POST",
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                // Update the midi_file widget with the new file path
                                const midiFileWidget = this.widgets.find(w => w.name === "midi_file");
                                if (midiFileWidget) {
                                    midiFileWidget.value = data.file_path;
                                    // Trigger any necessary updates
                                    if (midiFileWidget.callback) {
                                        midiFileWidget.callback(data.file_path);
                                    }
                                }
                                // You might want to call updateTrackSelection here if it's needed
                                this.updateTrackSelection();
                                this.saveState(); // Save state after successful upload
                            } else {
                                console.error("Failed to upload MIDI file:", data.error);
                            }
                        })
                        .catch(error => {
                            console.error("Error uploading MIDI file:", error);
                        });
                    }
                };
                input.click();
            };
            
            nodeType.prototype.onNodeCreated = function() {
                this.properties = this.properties || {};
                this.properties.selectedNotes = []; // Initialize with an empty array
                this.pianoScroll = 0;
                this.scrollLeftIndicator = null;
                this.scrollRightIndicator = null;
                this.availableNotes = new Set();

                this.addWidget("button", "Upload MIDI", "upload", () => {
                    this.uploadMIDI();
                });

                this.addWidget("button", "Refresh", "refresh", () => {
                    this.refreshMIDIData();
                });

                const notesWidget = this.widgets.find(widget => widget.name === "notes");
                if (notesWidget) {
                    // Disable the widget
                    this.disableWidget(notesWidget);
                    // Hide the widget
                    notesWidget.hidden = true;
                }

                // Add event listeners for midi_file and track_selection widgets
                const midiFileWidget = this.widgets.find(w => w.name === "midi_file");
                if (midiFileWidget) {
                    midiFileWidget.callback = (value) => {
                        this.updateTrackSelection();
                        this.saveState(); // Save state when MIDI file changes
                    };
                }

                const trackSelectionWidget = this.widgets.find(w => w.name === "track_selection");
                if (trackSelectionWidget) {
                    trackSelectionWidget.callback = () => this.onTrackSelectionChange(trackSelectionWidget.value);
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

                // Load saved state
                this.loadSavedState();

                // Set up a flag to indicate initial load
                this.isInitialLoad = true;

                // Instead of calling updateTrackSelection here, we'll set up a callback
                // that will be called when the graph is available
                this.onGraphAvailable = () => {
                    this.updateTrackSelection();
                };
            };

            nodeType.prototype.loadSavedState = function() {
                const savedState = localStorage.getItem(`MIDILoadAndExtract_${this.id}`);
                if (savedState) {
                    this.loadedState = JSON.parse(savedState);
                    
                    const midiFileWidget = this.widgets.find(w => w.name === "midi_file");
                    if (midiFileWidget && this.loadedState.midiFile) {
                        midiFileWidget.value = this.loadedState.midiFile;
                    }
                }
            };

            nodeType.prototype.updateTrackSelection = async function() {
                if (this.isUpdatingTrackSelection) {
                    console.log('Track selection update already in progress, skipping');
                    return;
                }
                this.isUpdatingTrackSelection = true;

                try {
                    const midiFile = this.getCurrentMidiFile();

                    if (!midiFile) {
                        console.warn('No MIDI file selected');
                        return;
                    }

                    const response = await fetch('/get_track_notes', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ midi_file: midiFile })
                    });
                    const data = await response.json();
                    
                    if (data.error) {
                        console.error('Error fetching track notes:', data.error);
                        return;
                    }
                    

                    const trackSelectionWidget = this.widgets.find(w => w.name === "track_selection");
                    if (trackSelectionWidget) {
                        // Update the options
                        trackSelectionWidget.options.values = data.tracks;
                    }

                    // Update available notes
                    this.availableNotes = new Set(data.all_notes.split(',').map(Number));
                    
                    // Apply loaded state if this is the initial load
                    if (this.isInitialLoad && this.loadedState) {
                        this.applyLoadedState();
                        this.isInitialLoad = false;
                    }

                    this.setDirtyCanvas(true, true);
                } catch (error) {
                    console.error('Error updating track selection:', error);
                    this.availableNotes = new Set(); // Reset to empty Set in case of error
                } finally {
                    this.isUpdatingTrackSelection = false;
                }
            };

            nodeType.prototype.refreshMIDIData = async function() {
                const midiFile = this.getCurrentMidiFile();
                const trackSelection = this.widgets.find(w => w.name === "track_selection").value;

                if (!midiFile) {
                    console.warn('No MIDI file selected');
                    return;
                }

                try {
                    const response = await fetch('/refresh_midi_data', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ midi_file: midiFile, track_selection: trackSelection })
                    });
                    const data = await response.json();
                    
                    if (data.error) {
                        console.error('Error refreshing MIDI data:', data.error);
                        return;
                    }
                    

                    // Update available tracks
                    const trackSelectionWidget = this.widgets.find(w => w.name === "track_selection");
                    if (trackSelectionWidget) {
                        trackSelectionWidget.options.values = data.tracks;
                    }

                    // Update available notes
                    this.availableNotes = new Set(data.all_notes.split(',').map(Number));
                    
                    // Filter out selected notes that are no longer available
                    if (this.properties.selectedNotes) {
                        this.properties.selectedNotes = this.properties.selectedNotes.filter(note => this.availableNotes.has(note));
                    } else {
                        this.properties.selectedNotes = [];
                    }
                    this.syncNotesWidget();

                    // Trigger redraw
                    this.setDirtyCanvas(true, true);
                } catch (error) {
                    console.error('Error refreshing MIDI data:', error);
                }
            };

            nodeType.prototype.applyLoadedState = function() {
                if (this.loadedState) {
                    
                    const trackSelectionWidget = this.widgets.find(w => w.name === "track_selection");
                    if (trackSelectionWidget && this.loadedState.trackSelection) {
                        if (trackSelectionWidget.options.values.includes(this.loadedState.trackSelection)) {
                            trackSelectionWidget.value = this.loadedState.trackSelection;
                        } else {
                            console.log(`Loaded track selection ${this.loadedState.trackSelection} is no longer valid`);
                        }
                    }
                    
                    if (this.loadedState.selectedNotes) {
                        this.properties.selectedNotes = this.loadedState.selectedNotes.filter(note => this.availableNotes.has(note));
                    }
                    
                    this.syncNotesWidget();
                    delete this.loadedState;
                    this.saveState(); // Save the applied state
                }
            };

            nodeType.prototype.saveState = function() {
                const state = {
                    midiFile: this.getCurrentMidiFile(),
                    trackSelection: this.widgets.find(w => w.name === "track_selection")?.value,
                    selectedNotes: this.properties.selectedNotes
                };
                localStorage.setItem(`MIDILoadAndExtract_${this.id}`, JSON.stringify(state));
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
                
                // Only call saveState if it's available
                if (typeof this.saveState === 'function') {
                    this.saveState();
                }
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

                // Ensure availableNotes exists before drawing
                if (!this.availableNotes) {
                    console.warn('availableNotes is undefined, initializing as empty Set');
                    this.availableNotes = new Set();
                }

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

            nodeType.prototype.getCurrentMidiFile = function() {
                const midiFileWidget = this.widgets.find(w => w.name === "midi_file");
                return midiFileWidget ? midiFileWidget.value : null;
            };

            nodeType.prototype.onTrackSelectionChange = async function(selectedTrack) {
                try {
                    const midiFile = this.getCurrentMidiFile();
                    const response = await fetch('/get_track_notes', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ midi_file: midiFile })
                    });
                    const data = await response.json();
                    
                    if (data.error) {
                        console.error('Error fetching track notes:', data.error);
                        return;
                    }
                    
                    const notesInTrack = selectedTrack === "all" 
                        ? data.all_notes 
                        : data.track_notes[selectedTrack.split(':')[0]];
                    
                    this.availableNotes = new Set(notesInTrack.split(',').map(Number));
                    
                    // Filter out selected notes that are no longer available
                    if (this.properties.selectedNotes) {
                        this.properties.selectedNotes = this.properties.selectedNotes.filter(note => this.availableNotes.has(note));
                    } else {
                        this.properties.selectedNotes = [];
                    }
                    this.syncNotesWidget();

                    // Trigger redraw
                    this.setDirtyCanvas(true, true);
                } catch (error) {
                    console.error('Error updating piano notes:', error);
                }
            };

            // Add this at the end of the function
            const onAdded = app.graph.onNodeAdded;
            app.graph.onNodeAdded = function(node) {
                if (onAdded) {
                    onAdded.call(this, node);
                }
                if (node.type === "MIDILoadAndExtract" && node.onGraphLoaded) {
                    node.onGraphLoaded();
                }
            };
        }
    },
});