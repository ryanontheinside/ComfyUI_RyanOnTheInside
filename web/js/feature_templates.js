// Feature template generation functions
export const FeatureTemplates = {
    // Basic waveforms
    sine: (frameCount, minValue, maxValue, cycles = 1) => {
        const amplitude = (maxValue - minValue) / 2;
        const center = minValue + amplitude;
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const value = center + amplitude * Math.sin((i / (numPoints - 1)) * Math.PI * 2 * cycles);
            points.push([frame, value]);
        }
        return points;
    },

    square: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const phase = (i / (numPoints - 1)) * cycles;
            const value = Math.floor(phase % 1 * 2) ? minValue : maxValue;
            points.push([frame, value]);
        }
        return points;
    },

    triangle: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const phase = (i / (numPoints - 1)) * cycles % 1;
            const triangleValue = phase < 0.5 
                ? 2 * phase 
                : 2 * (1 - phase);
            const value = minValue + (maxValue - minValue) * triangleValue;
            points.push([frame, value]);
        }
        return points;
    },

    sawtooth: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const phase = (i / (numPoints - 1)) * cycles % 1;
            const value = minValue + (maxValue - minValue) * phase;
            points.push([frame, value]);
        }
        return points;
    },

    // Animation curves
    easeInOut: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const t = (i / (numPoints - 1)) * cycles % 1;
            // Cubic ease in-out
            const easeValue = t < 0.5
                ? 4 * t * t * t
                : 1 - Math.pow(-2 * t + 2, 3) / 2;
            const value = minValue + (maxValue - minValue) * easeValue;
            points.push([frame, value]);
        }
        return points;
    },

    easeIn: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const t = (i / (numPoints - 1)) * cycles % 1;
            // Cubic ease in
            const easeValue = t * t * t;
            const value = minValue + (maxValue - minValue) * easeValue;
            points.push([frame, value]);
        }
        return points;
    },

    easeOut: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const t = (i / (numPoints - 1)) * cycles % 1;
            // Cubic ease out
            const easeValue = 1 - Math.pow(1 - t, 3);
            const value = minValue + (maxValue - minValue) * easeValue;
            points.push([frame, value]);
        }
        return points;
    },

    // Bounce patterns
    bounce: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const t = (i / (numPoints - 1)) * cycles % 1;
            // Bounce calculation with bounds check
            const bounceValue = Math.abs(Math.sin(t * Math.PI * 3) * (1 - t));
            const value = minValue + (maxValue - minValue) * (1 - bounceValue);
            points.push([frame, Math.max(minValue, Math.min(maxValue, value))]);
        }
        return points;
    },

    // Pulse patterns
    pulse: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const t = (i / (numPoints - 1)) * cycles % 1;
            // Gaussian pulse
            const pulseValue = Math.exp(-Math.pow((Math.sin(t * Math.PI * 2) * 2), 2));
            const value = minValue + (maxValue - minValue) * pulseValue;
            points.push([frame, Math.max(minValue, Math.min(maxValue, value))]);
        }
        return points;
    },

    // Random patterns
    random: (frameCount, minValue, maxValue) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const value = minValue + (maxValue - minValue) * Math.random();
            points.push([frame, value]);
        }
        return points;
    },

    smoothRandom: (frameCount, minValue, maxValue) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        let lastValue = minValue + (maxValue - minValue) * Math.random();
        const maxStep = (maxValue - minValue) * 0.2; // Maximum change per step
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const randomStep = (Math.random() * 2 - 1) * maxStep;
            lastValue = Math.max(minValue, Math.min(maxValue, lastValue + randomStep));
            points.push([frame, lastValue]);
        }
        return points;
    },

    // Special patterns
    heartbeat: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const t = (i / (numPoints - 1)) * cycles % 1;
            // Double-pulse heartbeat pattern with bounds check
            const x = t * 4;
            const heartbeatValue = Math.min(1, Math.max(0,
                Math.pow(Math.E, -Math.pow(x % 1 * 6 - 1, 2)) * 0.8 + 
                Math.pow(Math.E, -Math.pow(x % 1 * 6 - 2, 2))
            ));
            const value = minValue + (maxValue - minValue) * heartbeatValue;
            points.push([frame, value]);
        }
        return points;
    },

    steps: (frameCount, minValue, maxValue, cycles = 1) => {
        const points = [];
        const numPoints = Math.min(frameCount, 30);
        const numSteps = Math.max(2, Math.round(cycles * 5)); // 5 steps per cycle
        
        for (let i = 0; i < numPoints; i++) {
            const frame = Math.round((i / (numPoints - 1)) * (frameCount - 1));
            const t = (i / (numPoints - 1)) * cycles % 1;
            const step = Math.floor(t * numSteps) / (numSteps - 1);
            const value = minValue + (maxValue - minValue) * Math.min(1, Math.max(0, step));
            points.push([frame, value]);
        }
        return points;
    }
}; 