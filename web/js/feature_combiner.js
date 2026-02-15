import { app } from "../../scripts/app.js";

const LANE_COLORS = ["#4fc3f7", "#81c784", "#ffb74d"];

function buildAudioUrl(file) {
    if (!file || !file.filename) return null;
    const params = new URLSearchParams({
        filename: file.filename,
        type: file.type || "temp",
        subfolder: file.subfolder || "",
    });
    return `/view?${params.toString()}`;
}

app.registerExtension({
    name: "RyanOnTheInside.AdvancedFeatureCombiner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AdvancedFeatureCombiner") return;

        nodeType.size = [700, 650];

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        const onSerialize = nodeType.prototype.onSerialize;
        const onConfigure = nodeType.prototype.onConfigure;

        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const wdWidget = this.widgets?.find(w => w.name === "weight_data");
            if (wdWidget) wdWidget.hidden = true;

            this.lanePoints = {};
            this.waveformPeaks = [];
            this.audioDuration = 0;
            this.uiFrameCount = 0;
            this.connectedFeatures = [];
            this.featureData = {};
            this.combinedData = [];
            this.isDragging = false;
            this.selectedPoint = null;
            this.hoverPoint = null;
            this.activeLane = null;
            this.playheadNorm = 0;       // 0..1 synced scrub position
            this._audioUrl = null;
            this._rafId = null;

            try {
                const saved = JSON.parse(wdWidget?.value || "{}");
                if (typeof saved === "object" && !Array.isArray(saved)) {
                    this.lanePoints = saved;
                }
            } catch (e) { /* ignore */ }

            // --- DOM audio widget (native browser <audio> with scrub bar) ---
            this._createAudioWidget();

            this._detectConnectedFeatures();
            return r;
        };

        nodeType.prototype._createAudioWidget = function () {
            const container = document.createElement("div");
            container.style.cssText = "display:flex;flex-direction:column;gap:2px;padding:4px 8px;";

            const audio = document.createElement("audio");
            audio.controls = true;
            audio.style.cssText = "width:100%;height:36px;";
            container.appendChild(audio);

            this._audioEl = audio;
            this._audioContainer = container;

            // Sync playhead as user scrubs or audio plays
            const syncPlayhead = () => {
                const dur = audio.duration;
                if (dur && dur > 0) {
                    this.playheadNorm = audio.currentTime / dur;
                } else {
                    this.playheadNorm = 0;
                }
                this.setDirtyCanvas(true, true);
            };

            audio.addEventListener("timeupdate", syncPlayhead);
            audio.addEventListener("seeking", syncPlayhead);
            audio.addEventListener("seeked", syncPlayhead);
            audio.addEventListener("play", () => {
                // High-frequency sync during playback via rAF
                const tick = () => {
                    if (audio.paused) { this._rafId = null; return; }
                    syncPlayhead();
                    this._rafId = requestAnimationFrame(tick);
                };
                tick();
            });
            audio.addEventListener("pause", () => {
                if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
                syncPlayhead();
            });
            audio.addEventListener("ended", () => {
                this.playheadNorm = 0;
                this.setDirtyCanvas(true, true);
            });

            // Add as a DOM widget so LiteGraph manages its position
            const widget = this.addDOMWidget("audio_player", "custom", container, {
                serialize: false,
                hideOnZoom: false,
            });
            widget.computeSize = () => [this.size[0], 44];

            // Initially hidden until we get audio
            container.style.display = "none";
            this._audioWidget = widget;
        };

        nodeType.prototype.onSerialize = function (o) {
            onSerialize?.apply(this, arguments);
            o.lanePoints = this.lanePoints;
            o.uiFrameCount = this.uiFrameCount;
            o.waveformPeaks = this.waveformPeaks;
            o.audioDuration = this.audioDuration;
            o.featureData = this.featureData;
            o.combinedData = this.combinedData;
            o._audioUrl = this._audioUrl;
        };

        nodeType.prototype.onConfigure = function (o) {
            onConfigure?.apply(this, arguments);

            // Restore lane points: prefer node object, fall back to widget value
            if (o.lanePoints && Object.keys(o.lanePoints).length > 0) {
                this.lanePoints = o.lanePoints;
                this._syncWeightData();
            } else {
                // Try reading from the weight_data widget (workflow-level save)
                const wdWidget = this.widgets?.find(w => w.name === "weight_data");
                if (wdWidget?.value && wdWidget.value !== "{}") {
                    try {
                        const parsed = JSON.parse(wdWidget.value);
                        if (typeof parsed === "object" && !Array.isArray(parsed) && Object.keys(parsed).length > 0) {
                            this.lanePoints = parsed;
                        }
                    } catch (e) { /* ignore */ }
                }
            }

            if (o.uiFrameCount) this.uiFrameCount = o.uiFrameCount;
            if (o.waveformPeaks) this.waveformPeaks = o.waveformPeaks;
            if (o.audioDuration) this.audioDuration = o.audioDuration;
            if (o.featureData) this.featureData = o.featureData;
            if (o.combinedData) this.combinedData = o.combinedData;
            if (o._audioUrl) {
                this._audioUrl = o._audioUrl;
                if (this._audioEl) {
                    this._audioEl.src = o._audioUrl;
                    this._audioContainer.style.display = "";
                }
            }
            this._detectConnectedFeatures();
            // Ensure node is sized correctly for restored content
            setTimeout(() => this._autoFitHeight(), 0);
        };

        nodeType.prototype.onConnectionsChange = function () {
            this._detectConnectedFeatures();
            this.setDirtyCanvas(true, true);
        };

        nodeType.prototype.onExecuted = function (message) {
            if (message?.waveform_peaks) {
                const wp = message.waveform_peaks;
                this.waveformPeaks = Array.isArray(wp[0]) ? wp[0] : wp;
            }
            if (message?.audio_duration != null) {
                const ad = message.audio_duration;
                this.audioDuration = Array.isArray(ad) ? ad[0] : ad;
            }
            if (message?.frame_count != null) {
                const fc = message.frame_count;
                this.uiFrameCount = Array.isArray(fc) ? fc[0] : fc;
                this._ensureDefaultLanePoints(this.uiFrameCount);
            }
            if (message?.feature_data) {
                const fd = message.feature_data;
                this.featureData = Array.isArray(fd) ? fd[0] : fd;
            }
            if (message?.combined_data) {
                const cd = message.combined_data;
                this.combinedData = Array.isArray(cd) ? cd[0] : cd;
            }
            if (message?.audio_file) {
                const af = message.audio_file;
                const fileInfo = Array.isArray(af) ? af[0] : af;
                const url = buildAudioUrl(fileInfo);
                if (url && url !== this._audioUrl) {
                    this._audioUrl = url;
                    this._audioEl.src = url;
                    this._audioContainer.style.display = "";
                }
            }
            this._autoFitHeight();
            this.setDirtyCanvas(true, true);
        };

        nodeType.prototype.onRemoved = function () {
            if (this._rafId) cancelAnimationFrame(this._rafId);
            if (this._audioEl) { this._audioEl.pause(); this._audioEl.src = ""; }
        };

        // --- Helpers ---

        nodeType.prototype._detectConnectedFeatures = function () {
            this.connectedFeatures = [];
            if (!this.inputs) return;
            for (const input of this.inputs) {
                const m = input.name.match(/^feature_(\d+)$/);
                if (m && input.link != null) {
                    this.connectedFeatures.push(parseInt(m[1]));
                }
            }
        };

        // Initialize default endpoints or rescale existing ones when frame count becomes known
        nodeType.prototype._ensureDefaultLanePoints = function (fc) {
            if (fc <= 1) return;
            const lastFrame = fc - 1;
            let changed = false;
            for (const idx of this.connectedFeatures) {
                const pts = this.lanePoints[idx];
                if (!pts || pts.length === 0) {
                    // First time: create default endpoints at weight=1.0
                    this.lanePoints[idx] = [[0, 1.0], [lastFrame, 1.0]];
                    changed = true;
                } else if (pts.length === 2
                    && pts[0][0] === 0 && pts[0][1] === 1.0
                    && pts[1][1] === 1.0 && pts[1][0] !== lastFrame
                    && pts[1][0] <= 1) {
                    // Default endpoints were created with wrong frame count — fix the end
                    pts[1][0] = lastFrame;
                    changed = true;
                }
            }
            if (changed) this._syncWeightData();
        };

        nodeType.prototype._syncWeightData = function () {
            const wdWidget = this.widgets?.find(w => w.name === "weight_data");
            if (wdWidget) {
                wdWidget.value = JSON.stringify(this.lanePoints);
                if (this.onWidgetChanged) {
                    this.onWidgetChanged(wdWidget.name, wdWidget.value, wdWidget.value, wdWidget);
                }
            }
        };

        nodeType.prototype._getFrameCount = function () {
            if (this.uiFrameCount > 0) return this.uiFrameCount;
            const w = this.widgets?.find(w => w.name === "frame_count");
            if (w?.value > 0) return w.value;
            // Fallback: infer from max frame in lane points so restored points don't blow up
            let maxFrame = 0;
            for (const pts of Object.values(this.lanePoints || {})) {
                if (Array.isArray(pts)) {
                    for (const p of pts) maxFrame = Math.max(maxFrame, p[0]);
                }
            }
            return maxFrame > 0 ? maxFrame + 1 : 30;
        };

        nodeType.prototype._getWidgetAreaBottom = function () {
            const LG = window.LiteGraph || globalThis.LiteGraph;
            const SLOT_H = LG?.NODE_SLOT_HEIGHT ?? 20;
            const numInputs = (this.inputs || []).length;
            const numOutputs = (this.outputs || []).length;
            let y = Math.max(numInputs, numOutputs) * SLOT_H;
            for (const w of (this.widgets || [])) {
                if (!w.hidden) y += (w.computeSize?.(this.size[0])?.[1] ?? w.computeSize?.[1] ?? 20) + 4;
            }
            return y + 10;
        };

        nodeType.prototype._getLayout = function () {
            const M = 16;
            const widgetBottom = this._getWidgetAreaBottom();
            const topY = widgetBottom + 8;
            const graphW = this.size[0] - 2 * M;
            const hasWave = this.waveformPeaks.length > 0;
            const lanes = this.connectedFeatures;
            const numLanes = lanes.length;

            const WAVE_H = hasWave ? 40 : 0;
            const FEAT_H = 40;
            const WEIGHT_H = 60;
            const COMBINED_H = numLanes > 0 ? 50 : 0;
            const GAP = 3;

            // Total height needed below topY
            const contentH = WAVE_H + (WAVE_H > 0 ? GAP : 0)
                + numLanes * (FEAT_H + 1 + WEIGHT_H + GAP)
                + COMBINED_H + 10; // bottom padding

            return {
                M, topY, graphW, hasWave, lanes, numLanes,
                WAVE_H, FEAT_H, WEIGHT_H, COMBINED_H, GAP, contentH
            };
        };

        // Auto-resize node height to fit all content
        nodeType.prototype._autoFitHeight = function () {
            const L = this._getLayout();
            const neededH = L.topY + L.contentH;
            if (this.size[1] < neededH) {
                this.size[1] = neededH;
                this.setDirtyCanvas(true, true);
            }
        };

        // --- Drawing ---

        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags.collapsed) return;

            const L = this._getLayout();
            const { M, topY, graphW, hasWave, lanes, numLanes,
                    WAVE_H, FEAT_H, WEIGHT_H, COMBINED_H, GAP } = L;
            const fc = this._getFrameCount();
            if (fc <= 0 || graphW <= 0) return;

            let y = topY;
            const drawStartY = y; // for playhead

            // --- Waveform ---
            if (hasWave && WAVE_H > 0) {
                ctx.fillStyle = "#12141a";
                ctx.fillRect(M, y, graphW, WAVE_H);
                ctx.strokeStyle = "#2a2e3a";
                ctx.strokeRect(M, y, graphW, WAVE_H);

                const peaks = this.waveformPeaks;
                ctx.fillStyle = "#4477aa";
                const mid = y + WAVE_H / 2;
                const barW = Math.max(1, graphW / peaks.length);
                for (let i = 0; i < peaks.length; i++) {
                    const x = M + (i / peaks.length) * graphW;
                    const h = peaks[i] * (WAVE_H / 2) * 0.9;
                    ctx.fillRect(x, mid - h, barW, h * 2);
                }

                ctx.fillStyle = "#7799bb";
                ctx.font = "9px Arial";
                ctx.textAlign = "left";
                ctx.fillText("Waveform", M + 3, y + 10);
                y += WAVE_H + GAP;
            }

            this._weightLaneRects = {};

            // --- Per-feature: data display + weight lane ---
            for (let li = 0; li < numLanes; li++) {
                const featIdx = lanes[li];
                const color = LANE_COLORS[(featIdx - 1) % LANE_COLORS.length];

                // Feature data curve
                ctx.fillStyle = "#141414";
                ctx.fillRect(M, y, graphW, FEAT_H);
                ctx.strokeStyle = "#2a2a2a";
                ctx.strokeRect(M, y, graphW, FEAT_H);

                ctx.fillStyle = color;
                ctx.font = "10px Arial";
                ctx.textAlign = "left";
                ctx.fillText(`Feature ${featIdx}`, M + 4, y + 11);

                const fData = this.featureData[String(featIdx)];
                if (fData && fData.length > 0) {
                    const maxVal = Math.max(...fData.map(Math.abs), 0.001);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    for (let i = 0; i < fData.length; i++) {
                        const px = M + (i / (fData.length - 1)) * graphW;
                        const py = y + FEAT_H - (fData[i] / maxVal) * (FEAT_H - 14) - 2;
                        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                } else {
                    ctx.fillStyle = "#444";
                    ctx.font = "9px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("(execute to see data)", M + graphW / 2, y + FEAT_H / 2 + 3);
                }
                y += FEAT_H + 1;

                // Weight lane
                const wLaneY = y;
                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(M, wLaneY, graphW, WEIGHT_H);
                ctx.strokeStyle = "#333";
                ctx.strokeRect(M, wLaneY, graphW, WEIGHT_H);

                ctx.fillStyle = color + "88";
                ctx.font = "9px Arial";
                ctx.textAlign = "left";
                ctx.fillText(`Weight ${featIdx}`, M + 4, wLaneY + 10);

                ctx.fillStyle = "#444";
                ctx.font = "8px Arial";
                ctx.textAlign = "right";
                ctx.fillText("1", M - 3, wLaneY + 9);
                ctx.fillText("0", M - 3, wLaneY + WEIGHT_H - 1);

                // Grid
                ctx.strokeStyle = "#222";
                ctx.lineWidth = 0.5;
                const midY = wLaneY + WEIGHT_H / 2;
                ctx.beginPath();
                ctx.moveTo(M, midY);
                ctx.lineTo(M + graphW, midY);
                ctx.stroke();

                this._weightLaneRects[featIdx] = { y: wLaneY, h: WEIGHT_H };

                const points = this.lanePoints[featIdx] || [];
                const maxF = Math.max(fc - 1, 1);

                // Reference line at weight=1.0
                ctx.strokeStyle = color + "20";
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 4]);
                ctx.beginPath();
                ctx.moveTo(M, wLaneY);
                ctx.lineTo(M + graphW, wLaneY);
                ctx.stroke();
                ctx.setLineDash([]);

                if (points.length > 0) {
                    const sorted = [...points].sort((a, b) => a[0] - b[0]);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    for (let pi = 0; pi < sorted.length; pi++) {
                        const px = M + (sorted[pi][0] / maxF) * graphW;
                        const py = wLaneY + (1 - sorted[pi][1]) * WEIGHT_H;
                        if (pi === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                    }
                    ctx.stroke();

                    for (let pi = 0; pi < sorted.length; pi++) {
                        const px = M + (sorted[pi][0] / maxF) * graphW;
                        const py = wLaneY + (1 - sorted[pi][1]) * WEIGHT_H;
                        const isSel = this.selectedPoint?.lane === featIdx && this.selectedPoint?.index === pi;
                        const isHov = this.hoverPoint?.lane === featIdx && this.hoverPoint?.index === pi;
                        ctx.beginPath();
                        ctx.arc(px, py, isSel ? 6 : 4, 0, Math.PI * 2);
                        ctx.fillStyle = isSel ? "#fff" : (isHov ? "#ffff00" : color);
                        ctx.fill();
                        ctx.strokeStyle = "#000";
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                }

                y += WEIGHT_H + GAP;
            }

            // --- Combined (computed client-side from feature data + weight envelopes) ---
            let drawEndY = y;
            if (numLanes > 0 && COMBINED_H > 0) {
                ctx.fillStyle = "#111118";
                ctx.fillRect(M, y, graphW, COMBINED_H);
                ctx.strokeStyle = "#444";
                ctx.strokeRect(M, y, graphW, COMBINED_H);

                ctx.fillStyle = "#aaa";
                ctx.font = "10px Arial";
                ctx.textAlign = "left";
                ctx.fillText("Combined", M + 4, y + 12);

                const cd = this._computeCombinedPreview();
                if (cd && cd.length > 0) {
                    const maxVal = Math.max(...cd.map(Math.abs), 0.001);
                    ctx.strokeStyle = "#ccc";
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    for (let i = 0; i < cd.length; i++) {
                        const px = M + (i / (cd.length - 1)) * graphW;
                        const py = y + COMBINED_H - (cd[i] / maxVal) * (COMBINED_H - 14) - 2;
                        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                } else {
                    ctx.fillStyle = "#444";
                    ctx.font = "9px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("(execute to see combined preview)", M + graphW / 2, y + COMBINED_H / 2 + 3);
                }
                drawEndY = y + COMBINED_H;
            }

            // --- Playhead (vertical line synced across ALL lanes) ---
            if (this.playheadNorm > 0 && this.playheadNorm < 1) {
                const phX = M + this.playheadNorm * graphW;
                ctx.save();
                ctx.strokeStyle = "#ff4444";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(phX, drawStartY);
                ctx.lineTo(phX, drawEndY);
                ctx.stroke();

                // Small triangle indicator at top
                ctx.fillStyle = "#ff4444";
                ctx.beginPath();
                ctx.moveTo(phX - 4, drawStartY);
                ctx.lineTo(phX + 4, drawStartY);
                ctx.lineTo(phX, drawStartY + 6);
                ctx.closePath();
                ctx.fill();
                ctx.restore();
            }

            // Help text
            if (numLanes > 0) {
                ctx.fillStyle = "#444";
                ctx.font = "9px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Click weight lane to add points \u2022 Drag to move \u2022 Double-click to delete",
                    M + graphW / 2, drawStartY - 3);
            }
        };

        // --- Client-side combined preview ---
        // Interpolates weight envelopes and applies them to feature data
        // so the user sees the result instantly as they draw weights.

        nodeType.prototype._interpolateWeights = function (points, numSamples) {
            if (!points || points.length === 0) return null; // null = use default 1.0
            const sorted = [...points].sort((a, b) => a[0] - b[0]);
            const fc = this._getFrameCount();
            const maxF = Math.max(fc - 1, 1);
            const arr = new Float32Array(numSamples);
            for (let i = 0; i < numSamples; i++) {
                const frame = (i / (numSamples - 1)) * maxF;
                // Before first point
                if (frame <= sorted[0][0]) { arr[i] = sorted[0][1]; continue; }
                // After last point
                if (frame >= sorted[sorted.length - 1][0]) { arr[i] = sorted[sorted.length - 1][1]; continue; }
                // Between points - linear interp
                for (let p = 0; p < sorted.length - 1; p++) {
                    if (frame >= sorted[p][0] && frame <= sorted[p + 1][0]) {
                        const t = (frame - sorted[p][0]) / (sorted[p + 1][0] - sorted[p][0]);
                        arr[i] = sorted[p][1] + t * (sorted[p + 1][1] - sorted[p][1]);
                        break;
                    }
                }
            }
            return arr;
        };

        nodeType.prototype._computeCombinedPreview = function () {
            const lanes = this.connectedFeatures;
            if (lanes.length === 0) return null;

            // Check we have feature data from backend
            const hasData = lanes.some(idx => {
                const d = this.featureData[String(idx)];
                return d && d.length > 0;
            });
            if (!hasData) return null;

            // Use the length of the first available feature data as sample count
            const firstData = this.featureData[String(lanes[0])];
            if (!firstData) return null;
            const N = firstData.length;

            const mode = this.widgets?.find(w => w.name === "combine_mode")?.value || "weighted_sum";

            // Build per-feature weighted values
            const weightedArrays = [];
            const weightArrays = [];
            for (const idx of lanes) {
                const fData = this.featureData[String(idx)];
                if (!fData || fData.length === 0) continue;

                const weights = this._interpolateWeights(this.lanePoints[idx], N);
                const weighted = new Float32Array(N);
                const wArr = new Float32Array(N);
                for (let i = 0; i < N; i++) {
                    const w = weights ? weights[i] : 1.0;
                    // Resample feature data if lengths differ
                    const fi = Math.min(Math.round(i / (N - 1) * (fData.length - 1)), fData.length - 1);
                    weighted[i] = fData[fi] * w;
                    wArr[i] = w;
                }
                weightedArrays.push(weighted);
                weightArrays.push(wArr);
            }

            if (weightedArrays.length === 0) return null;

            const combined = new Float32Array(N);
            if (mode === "weighted_sum") {
                for (let i = 0; i < N; i++) {
                    let sum = 0;
                    for (const arr of weightedArrays) sum += arr[i];
                    combined[i] = sum;
                }
            } else if (mode === "normalized") {
                for (let i = 0; i < N; i++) {
                    let sum = 0, wsum = 0;
                    for (let j = 0; j < weightedArrays.length; j++) {
                        sum += weightedArrays[j][i];
                        wsum += weightArrays[j][i];
                    }
                    combined[i] = wsum > 0 ? sum / wsum : 0;
                }
            } else if (mode === "max") {
                for (let i = 0; i < N; i++) {
                    let mx = -Infinity;
                    for (const arr of weightedArrays) mx = Math.max(mx, arr[i]);
                    combined[i] = mx;
                }
            } else if (mode === "min") {
                for (let i = 0; i < N; i++) {
                    let mn = Infinity;
                    for (const arr of weightedArrays) mn = Math.min(mn, arr[i]);
                    combined[i] = mn;
                }
            } else if (mode === "multiply") {
                for (let i = 0; i < N; i++) {
                    let prod = 1;
                    for (const arr of weightedArrays) prod *= arr[i];
                    combined[i] = prod;
                }
            } else if (mode === "subtract") {
                for (let i = 0; i < N; i++) {
                    combined[i] = weightedArrays[0][i];
                    for (let j = 1; j < weightedArrays.length; j++) combined[i] -= weightedArrays[j][i];
                }
            } else if (mode === "divide") {
                for (let i = 0; i < N; i++) {
                    combined[i] = weightedArrays[0][i];
                    for (let j = 1; j < weightedArrays.length; j++) {
                        combined[i] = weightedArrays[j][i] !== 0 ? combined[i] / weightedArrays[j][i] : 0;
                    }
                }
            }
            return combined;
        };

        // --- Interaction (only weight lanes are interactive) ---

        nodeType.prototype._hitTestWeightLane = function (x, y) {
            const L = this._getLayout();
            if (x < L.M || x > L.M + L.graphW) return null;
            for (const [fidxStr, rect] of Object.entries(this._weightLaneRects || {})) {
                if (y >= rect.y && y < rect.y + rect.h) {
                    return { featIdx: parseInt(fidxStr), laneY: rect.y, laneH: rect.h };
                }
            }
            return null;
        };

        nodeType.prototype._posToLaneCoords = function (x, y, laneInfo) {
            const L = this._getLayout();
            const fc = this._getFrameCount();
            const frame = Math.round(((x - L.M) / L.graphW) * Math.max(fc - 1, 1));
            const weight = 1 - (y - laneInfo.laneY) / laneInfo.laneH;
            return [
                Math.max(0, Math.min(Math.max(fc - 1, 0), frame)),
                Math.max(0, Math.min(1, weight))
            ];
        };

        nodeType.prototype._findNearPointInLane = function (x, y, featIdx, laneInfo) {
            const L = this._getLayout();
            const fc = this._getFrameCount();
            const maxF = Math.max(fc - 1, 1);
            const points = this.lanePoints[featIdx] || [];
            for (let i = 0; i < points.length; i++) {
                const px = L.M + (points[i][0] / maxF) * L.graphW;
                const py = laneInfo.laneY + (1 - points[i][1]) * laneInfo.laneH;
                if (Math.sqrt((x - px) ** 2 + (y - py) ** 2) < 10) return i;
            }
            return null;
        };

        nodeType.prototype.onMouseDown = function (e, pos) {
            const [x, y] = pos;
            const lane = this._hitTestWeightLane(x, y);
            if (!lane) return false;

            const { featIdx } = lane;
            if (!this.lanePoints[featIdx]) this.lanePoints[featIdx] = [];

            const ptIdx = this._findNearPointInLane(x, y, featIdx, lane);
            if (ptIdx !== null) {
                this.selectedPoint = { lane: featIdx, index: ptIdx };
                this.isDragging = true;
                this.activeLane = lane;
            } else {
                const [frame, weight] = this._posToLaneCoords(x, y, lane);
                this.lanePoints[featIdx] = this.lanePoints[featIdx].filter(p => p[0] !== frame);
                this.lanePoints[featIdx].push([frame, weight]);
                this.lanePoints[featIdx].sort((a, b) => a[0] - b[0]);
                const idx = this.lanePoints[featIdx].findIndex(p => p[0] === frame);
                this.selectedPoint = { lane: featIdx, index: idx };
                this.isDragging = true;
                this.activeLane = lane;
                this._syncWeightData();
            }
            this.setDirtyCanvas(true, true);
            return true;
        };

        nodeType.prototype.onMouseMove = function (e, pos) {
            const [x, y] = pos;
            if (this.isDragging && this.selectedPoint && this.activeLane) {
                const { lane: featIdx, index } = this.selectedPoint;
                const [frame, weight] = this._posToLaneCoords(x, y, this.activeLane);
                const points = this.lanePoints[featIdx];
                if (points && points[index]) {
                    const dup = points.findIndex((p, i) => i !== index && p[0] === frame);
                    if (dup === -1) {
                        points[index] = [frame, weight];
                        points.sort((a, b) => a[0] - b[0]);
                        this.selectedPoint.index = points.findIndex(p => p[0] === frame && p[1] === weight);
                        this._syncWeightData();
                    }
                }
                this.setDirtyCanvas(true, true);
                return true;
            }
            const lane = this._hitTestWeightLane(x, y);
            if (lane) {
                const ptIdx = this._findNearPointInLane(x, y, lane.featIdx, lane);
                this.hoverPoint = ptIdx !== null ? { lane: lane.featIdx, index: ptIdx } : null;
            } else {
                this.hoverPoint = null;
            }
            this.setDirtyCanvas(true, true);
            return false;
        };

        nodeType.prototype.onMouseUp = function () {
            if (this.isDragging) {
                this.isDragging = false;
                this.selectedPoint = null;
                this.activeLane = null;
                this.setDirtyCanvas(true, true);
            }
            return false;
        };

        nodeType.prototype.onDblClick = function (e, pos) {
            const [x, y] = pos;
            const lane = this._hitTestWeightLane(x, y);
            if (!lane) return false;
            const { featIdx } = lane;
            const ptIdx = this._findNearPointInLane(x, y, featIdx, lane);
            if (ptIdx !== null) {
                this.lanePoints[featIdx].splice(ptIdx, 1);
                this.selectedPoint = null;
                this.isDragging = false;
                this._syncWeightData();
                this.setDirtyCanvas(true, true);
                return true;
            }
            return false;
        };

        nodeType.prototype.onResize = function (size) {
            this.size[0] = Math.max(500, size[0]);
            this.size[1] = Math.max(400, size[1]);
            this.setDirtyCanvas(true, true);
        };
    }
});
