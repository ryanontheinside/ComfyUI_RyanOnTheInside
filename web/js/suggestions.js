import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "RyanOnTheInside.Suggestions",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Get access to the suggestion lists via the SlotDefaults extension
        const suggestions = app.extensions.find(ext => ext.name === "Comfy.SlotDefaults");
        
        if (suggestions) {
            // console.log("[RyanOnTheInside.Suggestions] Initializing feature suggestions");
            
            const type = "FEATURE";  // Our specific type
            
            // Override input suggestions for the type
            suggestions.slot_types_default_in[type] = [
                "Reroute",  // Usually keep Reroute as first option
                "AudioFeatureExtractor",
                "MIDIFeatureExtractor",
                "MotionFeatureNode",
                "WhisperFeatureExtractor",
                "FeatureMixer"
            ];
            
            // Override output suggestions for the type
            suggestions.slot_types_default_out[type] = [
                "Reroute",  // Usually keep Reroute as first option
                "PreviewFeature",
                "FeatureInfoNode",
                "FeatureToFlexIntParam",
                "FeatureToFlexFloatParam"
            ];

            // Register with LiteGraph's type system
            const lowerType = type.toLowerCase();
            
            // Register input type if not already registered
            if (!(lowerType in LiteGraph.registered_slot_in_types)) {
                LiteGraph.registered_slot_in_types[lowerType] = { nodes: [] };
            }
            // Add the node classes that can accept this type as input
            suggestions.slot_types_default_in[type].forEach(nodeId => {
                const nodeType = LiteGraph.registered_node_types[nodeId];
                if (nodeType?.comfyClass) {
                    LiteGraph.registered_slot_in_types[lowerType].nodes.push(nodeType.comfyClass);
                }
            });

            // Register output type if not already registered
            if (!(type in LiteGraph.registered_slot_out_types)) {
                LiteGraph.registered_slot_out_types[type] = { nodes: [] };
            }
            // Add the node classes that can output this type
            suggestions.slot_types_default_out[type].forEach(nodeId => {
                const nodeType = LiteGraph.registered_node_types[nodeId];
                if (nodeType?.comfyClass) {
                    LiteGraph.registered_slot_out_types[type].nodes.push(nodeType.comfyClass);
                }
            });

            // Register as valid output type if not already registered
            if (!LiteGraph.slot_types_out.includes(type)) {
                LiteGraph.slot_types_out.push(type);
            }

            // console.log("[RyanOnTheInside.Suggestions] Feature suggestions initialized successfully");
        } else {
            console.warn("[RyanOnTheInside.Suggestions] SlotDefaults extension not found");
        }
    }
});