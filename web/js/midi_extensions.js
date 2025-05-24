import { app } from "../../scripts/app.js";
import { drawPiano, handlePianoMouseDown } from "./piano.js";

app.registerExtension({
    name: "org.ryanontheinside.midi_feature_extractor",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        //claude: do not change the node name
        if (nodeData.name === "MIDIFeatureExtractor") {

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
                this.properties = this.properties || {};
                this.properties.selectedNotes = []; // Initialize with an empty array
                this.pianoScroll = 0;
                this.scrollLeftIndicator = null;
                this.scrollRightIndicator = null;
                this.availableNotes = new Set();

                const notesWidget = this.widgets.find(widget => widget.name === "notes");
                if (notesWidget) {
                    // Disable the widget
                    this.disableWidget(notesWidget);
                    // Hide the widget
                    notesWidget.hidden = true;
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

                // Check for existing connections after a short delay
                setTimeout(() => {
                    const connectedNodes = this.graph._nodes.filter(
                        n => n.type === 'MIDILoader' && this.isNodeConnected(n)
                    );
                    
                    if (connectedNodes.length > 0) {
                        const loaderNode = connectedNodes[0];
                        this.queryMIDILoaderNotes(loaderNode);
                        this.listenToMIDILoaderChanges(loaderNode);
                    }
                }, 100); // Small delay to ensure links are established
            };

            // Handle connections changing
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                if (type === 1) { // Input connection (1 is for inputs, 0 is for outputs)
                    if (connected && link_info) {
                        // New connection established
                        const inputNode = this.graph.getNodeById(link_info.origin_id);
                        if (inputNode && inputNode.type === 'MIDILoader') {
                            this.queryMIDILoaderNotes(inputNode);
                            this.listenToMIDILoaderChanges(inputNode);
                        }
                    } else {
                        // Check if we still have any connected MIDILoader nodes
                        const connectedMidiNodes = this.graph._nodes.filter(
                            n => n.type === 'MIDILoader' && this.isNodeConnected(n)
                        );
                        
                        if (connectedMidiNodes.length === 0) {
                            // All MIDILoader connections removed
                            this.availableNotes = new Set();
                            this.properties.selectedNotes = [];
                            this.syncNotesWidget();
                            this.setDirtyCanvas(true, true);
                        }
                    }
                }
            };

            // Check if a node is connected to this node
            nodeType.prototype.isNodeConnected = function(node) {
                for (const link_id in this.graph.links) {
                    const link = this.graph.links[link_id];
                    if (link.target_id === this.id && link.origin_id === node.id) {
                        return true;
                    }
                }
                return false;
            };

            // Query the MIDILoader for available notes
            nodeType.prototype.queryMIDILoaderNotes = async function(loaderNode) {
                if (!loaderNode) return;
                
                const midiFile = loaderNode.getCurrentMidiFile();
                const trackSelection = loaderNode.widgets.find(w => w.name === "track_selection").value;
                
                // Get measure selection parameters from the MIDILoader
                const startMeasureWidget = loaderNode.widgets.find(w => w.name === "start_measure");
                const startBeatWidget = loaderNode.widgets.find(w => w.name === "start_beat");
                const endMeasureWidget = loaderNode.widgets.find(w => w.name === "end_measure");
                const endBeatWidget = loaderNode.widgets.find(w => w.name === "end_beat");
                
                const startMeasure = startMeasureWidget ? startMeasureWidget.value : 1;
                const startBeat = startBeatWidget ? startBeatWidget.value : 1;
                const endMeasure = endMeasureWidget ? endMeasureWidget.value : 0;
                const endBeat = endBeatWidget ? endBeatWidget.value : 1;
                
                if (!midiFile) return;
                
                try {
                    console.log(`Querying MIDI notes with startMeasure=${startMeasure}, startBeat=${startBeat}, endMeasure=${endMeasure}, endBeat=${endBeat}`);
                    
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
                        console.error('Error fetching MIDI notes:', data.error);
                        return;
                    }
                    
                    // Update available notes
                    const noteArray = data.all_notes.split(',').map(Number).filter(n => !isNaN(n));
                    console.log(`Received ${noteArray.length} available notes from measures ${startMeasure} to ${endMeasure}, time signature: ${data.time_signature}`);
                    
                    this.availableNotes = new Set(noteArray);
                    
                    // Filter out selected notes that are no longer available
                    if (this.properties.selectedNotes) {
                        const originalCount = this.properties.selectedNotes.length;
                        this.properties.selectedNotes = this.properties.selectedNotes.filter(
                            note => this.availableNotes.has(note)
                        );
                        
                        if (originalCount !== this.properties.selectedNotes.length) {
                            console.log(`Filtered out ${originalCount - this.properties.selectedNotes.length} selected notes`);
                        }
                    }
                    
                    this.syncNotesWidget();
                    this.setDirtyCanvas(true, true);
                } catch (error) {
                    console.error('Error querying MIDI loader:', error);
                }
            };

            // Listen for changes in the MIDILoader node
            nodeType.prototype.listenToMIDILoaderChanges = function(loaderNode) {
                // Store reference to loader node
                this.connectedLoaderNode = loaderNode;
                
                // Check if we already added a listener to this node
                if (loaderNode._feature_extractor_listeners && 
                    loaderNode._feature_extractor_listeners.includes(this.id)) {
                    return;
                }
                
                // Initialize listener array if needed
                if (!loaderNode._feature_extractor_listeners) {
                    loaderNode._feature_extractor_listeners = [];
                }
                loaderNode._feature_extractor_listeners.push(this.id);
                
                // Find all relevant widgets in loader node
                const trackWidget = loaderNode.widgets.find(w => w.name === "track_selection");
                const startMeasureWidget = loaderNode.widgets.find(w => w.name === "start_measure");
                const startBeatWidget = loaderNode.widgets.find(w => w.name === "start_beat");
                const endMeasureWidget = loaderNode.widgets.find(w => w.name === "end_measure");
                const endBeatWidget = loaderNode.widgets.find(w => w.name === "end_beat");
                
                // Set up callbacks for all relevant widgets
                if (trackWidget) {
                    const origTrackCallback = trackWidget.callback;
                    trackWidget.callback = (value) => {
                        // Call original callback
                        if (origTrackCallback) origTrackCallback.call(loaderNode, value);
                        
                        // Notify this feature extractor
                        this.queryMIDILoaderNotes(loaderNode);
                    };
                }
                
                if (startMeasureWidget) {
                    const origCallback = startMeasureWidget.callback;
                    startMeasureWidget.callback = (value) => {
                        // Call original callback
                        if (origCallback) origCallback.call(loaderNode, value);
                        
                        // Notify this feature extractor
                        this.queryMIDILoaderNotes(loaderNode);
                    };
                }
                
                if (startBeatWidget) {
                    const origCallback = startBeatWidget.callback;
                    startBeatWidget.callback = (value) => {
                        // Call original callback
                        if (origCallback) origCallback.call(loaderNode, value);
                        
                        // Notify this feature extractor
                        this.queryMIDILoaderNotes(loaderNode);
                    };
                }
                
                if (endMeasureWidget) {
                    const origCallback = endMeasureWidget.callback;
                    endMeasureWidget.callback = (value) => {
                        // Call original callback
                        if (origCallback) origCallback.call(loaderNode, value);
                        
                        // Notify this feature extractor
                        this.queryMIDILoaderNotes(loaderNode);
                    };
                }
                
                if (endBeatWidget) {
                    const origCallback = endBeatWidget.callback;
                    endBeatWidget.callback = (value) => {
                        // Call original callback
                        if (origCallback) origCallback.call(loaderNode, value);
                        
                        // Notify this feature extractor
                        this.queryMIDILoaderNotes(loaderNode);
                    };
                }
            };

            nodeType.prototype.loadSavedState = function() {
                const savedState = localStorage.getItem(`MIDIFeatureExtractor_${this.id}`);
                if (savedState) {
                    try {
                        const state = JSON.parse(savedState);
                        if (state.selectedNotes) {
                            this.properties.selectedNotes = state.selectedNotes;
                        }
                    } catch (e) {
                        console.error("Error loading saved state:", e);
                    }
                }
            };

            nodeType.prototype.saveState = function() {
                const state = {
                    selectedNotes: this.properties.selectedNotes
                };
                localStorage.setItem(`MIDIFeatureExtractor_${this.id}`, JSON.stringify(state));
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
                
                // Save state
                this.saveState();
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
        }
    },
});