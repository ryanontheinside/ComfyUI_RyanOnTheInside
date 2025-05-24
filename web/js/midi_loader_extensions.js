import { app } from "../../scripts/app.js";
import { drawPiano, handlePianoMouseDown } from "./piano.js";

app.registerExtension({
    name: "org.ryanontheinside.midi_loader",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MIDILoader") {

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
                            if (data.status === "success") {
                                // Update the midi_file widget with the new file
                                const midiFileWidget = this.widgets.find(w => w.name === "midi_file");
                                if (midiFileWidget) {
                                    midiFileWidget.value = data.uploaded_file;
                                    // Trigger any necessary updates
                                    if (midiFileWidget.callback) {
                                        midiFileWidget.callback(data.uploaded_file);
                                    }
                                }
                                this.updateTrackSelection();
                                this.saveState(); 
                            } else {
                                console.error("Failed to upload MIDI file:", data.message);
                            }
                        })
                        .catch(error => {
                            console.error("Error uploading MIDI file:", error);
                        });
                    }
                };
                input.click();
            };
            
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                this.properties = this.properties || {};
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

                // Add event listeners for midi_file widget
                const midiFileWidget = this.widgets.find(w => w.name === "midi_file");
                if (midiFileWidget) {
                    midiFileWidget.callback = (value) => {
                        this.updateTrackSelection();
                        this.saveState(); // Save state when MIDI file changes
                    };
                }

                const trackSelectionWidget = this.widgets.find(w => w.name === "track_selection");
                if (trackSelectionWidget) {
                    const origTrackCallback = trackSelectionWidget.callback;
                    trackSelectionWidget.callback = (value) => {
                        // Call original callback if it exists
                        if (origTrackCallback) origTrackCallback.call(this, value);
                        this.saveState();
                        
                        // Notify connected nodes of the change
                        this.notifyConnectedNodes();
                    };
                }
                
                // Set up callbacks for start time and duration widgets
                const startTimeWidget = this.widgets.find(w => w.name === "start_time_seconds");
                if (startTimeWidget) {
                    const origStartCallback = startTimeWidget.callback;
                    startTimeWidget.callback = (value) => {
                        // Call original callback if it exists
                        if (origStartCallback) origStartCallback.call(this, value);
                        this.saveState();
                        
                        // Notify connected nodes of the change
                        this.notifyConnectedNodes();
                    };
                }
                
                const durationWidget = this.widgets.find(w => w.name === "duration_seconds");
                if (durationWidget) {
                    const origDurCallback = durationWidget.callback;
                    durationWidget.callback = (value) => {
                        // Call original callback if it exists
                        if (origDurCallback) origDurCallback.call(this, value);
                        this.saveState();
                        
                        // Notify connected nodes of the change
                        this.notifyConnectedNodes();
                    };
                }

                // Increase the node's height and width
                const extraHeight = 80; // Piano height + margins
                const pianoWidth = 380; // Reduced width for just the loader
                this.size[1] += extraHeight;
                this.size[0] = Math.max(this.size[0], pianoWidth);

                // Ensure the node recomputes its size
                this.setDirtyCanvas(true, true);

                // Load saved state
                this.loadSavedState();

                // Set up a flag to indicate initial load
                this.isInitialLoad = true;

                // Set up a callback that will be called when the graph is available
                this.onGraphAvailable = () => {
                    this.updateTrackSelection();
                };
                
                // Initialize the feature extractor listeners array
                this._feature_extractor_listeners = [];
                
                // Same for refreshMIDIData method
                const originalRefreshMIDIData = this.refreshMIDIData;
                this.refreshMIDIData = async function() {
                    await originalRefreshMIDIData.call(this);
                    this.notifyConnectedNodes();
                };
            };

            nodeType.prototype.loadSavedState = function() {
                const savedState = localStorage.getItem(`MIDILoader_${this.id}`);
                if (savedState) {
                    this.loadedState = JSON.parse(savedState);
                    
                    const midiFileWidget = this.widgets.find(w => w.name === "midi_file");
                    if (midiFileWidget && this.loadedState.midiFile) {
                        midiFileWidget.value = this.loadedState.midiFile;
                    }
                    
                    if (this.loadedState.trackSelection) {
                        const trackWidget = this.widgets.find(w => w.name === "track_selection");
                        if (trackWidget) {
                            trackWidget.value = this.loadedState.trackSelection;
                        }
                    }
                    
                    // Restore measure/beat selection
                    if (this.loadedState.startMeasure !== undefined) {
                        const widget = this.widgets.find(w => w.name === "start_measure");
                        if (widget) widget.value = this.loadedState.startMeasure;
                    }
                    
                    if (this.loadedState.startBeat !== undefined) {
                        const widget = this.widgets.find(w => w.name === "start_beat");
                        if (widget) widget.value = this.loadedState.startBeat;
                    }
                    
                    if (this.loadedState.endMeasure !== undefined) {
                        const widget = this.widgets.find(w => w.name === "end_measure");
                        if (widget) widget.value = this.loadedState.endMeasure;
                    }
                    
                    if (this.loadedState.endBeat !== undefined) {
                        const widget = this.widgets.find(w => w.name === "end_beat");
                        if (widget) widget.value = this.loadedState.endBeat;
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
                    
                    // Get measure selection parameters
                    const startMeasureWidget = this.widgets.find(w => w.name === "start_measure");
                    const startBeatWidget = this.widgets.find(w => w.name === "start_beat");
                    const endMeasureWidget = this.widgets.find(w => w.name === "end_measure");
                    const endBeatWidget = this.widgets.find(w => w.name === "end_beat");
                    
                    const startMeasure = startMeasureWidget ? startMeasureWidget.value : 1;
                    const startBeat = startBeatWidget ? startBeatWidget.value : 1;
                    const endMeasure = endMeasureWidget ? endMeasureWidget.value : 0;
                    const endBeat = endBeatWidget ? endBeatWidget.value : 1;

                    const response = await fetch('/get_track_notes', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            midi_file: midiFile,
                            start_measure: startMeasure,
                            start_beat: startBeat,
                            end_measure: endMeasure,
                            end_beat: endBeat
                        })
                    });
                    const data = await response.json();
                    
                    if (data.error) {
                        console.error('Error fetching track notes:', data.error);
                        return;
                    }
                    
                    const trackWidget = this.widgets.find(w => w.name === "track_selection");
                    if (trackWidget) {
                        // Update the options
                        trackWidget.options.values = data.tracks;
                    }
                    
                    // Apply loaded state if this is the initial load
                    if (this.isInitialLoad && this.loadedState) {
                        if (this.loadedState.trackSelection) {
                            const trackWidget = this.widgets.find(w => w.name === "track_selection");
                            if (trackWidget && trackWidget.options.values.includes(this.loadedState.trackSelection)) {
                                trackWidget.value = this.loadedState.trackSelection;
                            }
                        }
                        
                        this.isInitialLoad = false;
                    }

                    this.setDirtyCanvas(true, true);
                    
                    // Notify connected nodes after track selection update
                    this.notifyConnectedNodes();
                } catch (error) {
                    console.error('Error updating track selection:', error);
                } finally {
                    this.isUpdatingTrackSelection = false;
                }
            };

            nodeType.prototype.refreshMIDIData = async function() {
                const midiFile = this.getCurrentMidiFile();
                const trackWidget = this.widgets.find(w => w.name === "track_selection");
                const trackSelection = trackWidget ? trackWidget.value : "all";
                
                // Get measure selection parameters
                const startMeasureWidget = this.widgets.find(w => w.name === "start_measure");
                const startBeatWidget = this.widgets.find(w => w.name === "start_beat");
                const endMeasureWidget = this.widgets.find(w => w.name === "end_measure");
                const endBeatWidget = this.widgets.find(w => w.name === "end_beat");
                
                const startMeasure = startMeasureWidget ? startMeasureWidget.value : 1;
                const startBeat = startBeatWidget ? startBeatWidget.value : 1;
                const endMeasure = endMeasureWidget ? endMeasureWidget.value : 0;
                const endBeat = endBeatWidget ? endBeatWidget.value : 1;

                if (!midiFile) {
                    console.warn('No MIDI file selected');
                    return;
                }

                try {
                    const response = await fetch('/refresh_midi_data', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            midi_file: midiFile, 
                            track_selection: trackSelection,
                            start_measure: startMeasure,
                            start_beat: startBeat,
                            end_measure: endMeasure,
                            end_beat: endBeat
                        })
                    });
                    const data = await response.json();
                    
                    if (data.error) {
                        console.error('Error refreshing MIDI data:', data.error);
                        return;
                    }
                    
                    // Update available tracks
                    const trackWidget = this.widgets.find(w => w.name === "track_selection");
                    if (trackWidget) {
                        trackWidget.options.values = data.tracks;
                    }

                    // Trigger redraw
                    this.setDirtyCanvas(true, true);
                } catch (error) {
                    console.error('Error refreshing MIDI data:', error);
                }
            };

            nodeType.prototype.saveState = function() {
                // Get current widgets' values
                const startMeasureWidget = this.widgets.find(w => w.name === "start_measure");
                const startBeatWidget = this.widgets.find(w => w.name === "start_beat");
                const endMeasureWidget = this.widgets.find(w => w.name === "end_measure");
                const endBeatWidget = this.widgets.find(w => w.name === "end_beat");
                
                const state = {
                    midiFile: this.getCurrentMidiFile(),
                    trackSelection: this.widgets.find(w => w.name === "track_selection")?.value,
                    startMeasure: startMeasureWidget ? startMeasureWidget.value : 1,
                    startBeat: startBeatWidget ? startBeatWidget.value : 1,
                    endMeasure: endMeasureWidget ? endMeasureWidget.value : 0,
                    endBeat: endBeatWidget ? endBeatWidget.value : 1
                };
                localStorage.setItem(`MIDILoader_${this.id}`, JSON.stringify(state));
            };

            nodeType.prototype.getCurrentMidiFile = function() {
                const midiFileWidget = this.widgets.find(w => w.name === "midi_file");
                return midiFileWidget ? midiFileWidget.value : null;
            };
            
            // Add a method to notify connected feature extractors
            nodeType.prototype.notifyConnectedNodes = function() {
                if (!this._feature_extractor_listeners) return;
                
                for (const id of this._feature_extractor_listeners) {
                    const node = this.graph.getNodeById(id);
                    if (node && node.queryMIDILoaderNotes) {
                        node.queryMIDILoaderNotes(this);
                    }
                }
            };

            // Handle disconnection to clean up listeners
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                if (type === 0 && !connected) { // Output disconnected
                    // Clean up the disconnected listener
                    if (link_info && this._feature_extractor_listeners) {
                        const targetNode = this.graph.getNodeById(link_info.target_id);
                        if (targetNode) {
                            this._feature_extractor_listeners = this._feature_extractor_listeners.filter(
                                id => id !== targetNode.id
                            );
                        }
                    }
                }
            };

            // Add this at the end of the function
            const onAdded = app.graph.onNodeAdded;
            app.graph.onNodeAdded = function(node) {
                if (onAdded) {
                    onAdded.call(this, node);
                }
                if (node.type === "MIDILoader" && node.onGraphLoaded) {
                    node.onGraphLoaded();
                }
            };
        }
    },
}); 