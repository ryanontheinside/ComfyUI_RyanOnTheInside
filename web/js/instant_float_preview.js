import { app } from "../../../scripts/app.js";

const MIN_GRAPH_HEIGHT = 500;
const AUDIO_HEIGHT = 44;
const LOOP_BAR_H = 18;
const HANDLE_W = 6;

function buildAudioUrl(file) {
    const params = new URLSearchParams({
        filename: file.filename,
        type: file.type,
        subfolder: file.subfolder || "",
    });
    return `/view?${params.toString()}`;
}

app.registerExtension({
    name: "RyanOnTheInside.InstantFloatPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "InstantFloatPreview") return;

        // --- Serialization: persist loop state across page refresh ---
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            onSerialize?.apply(this, arguments);
            if (this._ifpLoop) {
                data.ifpLoop = { ...this._ifpLoop };
            }
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            onConfigure?.apply(this, arguments);
            if (data.ifpLoop) {
                const saved = data.ifpLoop;
                if (this._ifpLoop) {
                    this._ifpLoop = saved;
                } else {
                    // Widget not created yet; stash for _ifpCreateWidget
                    this._ifpPendingLoop = saved;
                }
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            // Suppress any default image preview/thumbnail
            if (this.imgs) this.imgs = null;
            if (this.imageIndex != null) this.imageIndex = null;

            const series = message?.series;
            const title = message?.title?.[0] || "Float Preview";
            const rewindSeconds = message?.rewind_seconds?.[0] ?? 2.0;
            const audioFile = message?.ifp_audio?.[0];
            const audioDuration = message?.ifp_audio_duration?.[0];

            if (!series || series.length === 0) return;

            if (!this._ifpContainer) {
                this._ifpCreateWidget();
            }

            this._ifpData = { series, title };

            // Handle audio
            if (audioFile) {
                this._ifpHandleAudio(audioFile, audioDuration, rewindSeconds);
            } else if (this._ifpAudioShown) {
                this._ifpAudioWrapper.style.display = "none";
                this._ifpAudioShown = false;
                this._ifpAudioDuration = 0;
            }

            // Draw after a frame so canvas is laid out
            requestAnimationFrame(() => {
                this._ifpDraw();
                if (!this._ifpSized) {
                    this._ifpSized = true;
                    const sz = this.computeSize();
                    sz[0] = Math.max(sz[0], 1680);
                    this.setSize(sz);
                    this.graph?.setDirtyCanvas(true, true);
                }
            });
        };

        nodeType.prototype._ifpCreateWidget = function () {
            const container = document.createElement("div");
            container.style.cssText = "display:flex;flex-direction:column;gap:0;min-height:" + MIN_GRAPH_HEIGHT + "px;";

            const canvas = document.createElement("canvas");
            canvas.style.cssText = "width:100%;flex:1;display:block;cursor:default;min-height:" + MIN_GRAPH_HEIGHT + "px;";
            container.appendChild(canvas);

            // Double-buffered audio: two elements, only the active one is visible.
            // This allows seamless audio swaps: new audio loads in the background
            // element while the old one keeps playing.
            const audioWrapper = document.createElement("div");
            audioWrapper.style.cssText = "width:100%;display:none;flex-shrink:0;";
            container.appendChild(audioWrapper);

            const audioA = document.createElement("audio");
            audioA.controls = true;
            audioA.preload = "auto";
            audioA.style.cssText = "width:100%;display:block;";
            audioWrapper.appendChild(audioA);

            const audioB = document.createElement("audio");
            audioB.controls = true;
            audioB.preload = "auto";
            audioB.style.cssText = "width:100%;display:none;";
            audioWrapper.appendChild(audioB);

            this._ifpContainer = container;
            this._ifpCanvas = canvas;
            this._ifpAudioWrapper = audioWrapper;
            this._ifpAudioA = audioA;
            this._ifpAudioB = audioB;
            this._ifpActiveAudio = audioA;
            this._ifpInactiveAudio = audioB;
            this._ifpData = null;
            this._ifpRafId = null;
            this._ifpAudioShown = false;
            this._ifpAudioDuration = 0;
            this._ifpSized = false;

            // Loop state
            this._ifpLoop = { enabled: false, start: 0, end: 1 };
            // Restore saved loop state from onConfigure if it ran before widget creation
            if (this._ifpPendingLoop) {
                this._ifpLoop = this._ifpPendingLoop;
                this._ifpPendingLoop = null;
            }
            this._ifpDrag = null;
            this._ifpSwapping = false;
            this._ifpRealTime = 0;
            this._ifpRealPlaying = false;
            this._ifpSwapAbort = null;

            const self = this;
            const widget = this.addDOMWidget("instant_float_preview", "custom", container, {
                serialize: false,
                hideOnZoom: false,
            });
            widget.computeSize = function () {
                const audioH = self._ifpAudioShown ? AUDIO_HEIGHT : 0;
                return [self.size[0], MIN_GRAPH_HEIGHT + audioH + 4];
            };

            // Redraw when the canvas element is resized (handles node resize + recovery from drag)
            this._ifpResizeObserver = new ResizeObserver(() => {
                this._ifpDraw();
            });
            this._ifpResizeObserver.observe(canvas);

            // Canvas mouse events for loop bar
            canvas.addEventListener("mousedown", (e) => this._ifpOnMouseDown(e));
            canvas.addEventListener("mousemove", (e) => this._ifpOnMouseMove(e));
            canvas.addEventListener("mouseup", (e) => this._ifpOnMouseUp(e));
            canvas.addEventListener("mouseleave", (e) => this._ifpOnMouseUp(e));
            canvas.addEventListener("dblclick", (e) => this._ifpOnDblClick(e));

            // Audio event listeners on both elements; only respond when the
            // element is the active one and we're not mid-swap.
            const setupAudioEvents = (el) => {
                el.addEventListener("timeupdate", () => {
                    if (el !== this._ifpActiveAudio || this._ifpSwapping) return;
                    this._ifpRealTime = el.currentTime;
                    this._ifpCheckLoop();
                    if (!this._ifpRafId) this._ifpDraw();
                });
                el.addEventListener("play", () => {
                    if (el !== this._ifpActiveAudio) return;
                    if (!this._ifpSwapping) this._ifpRealPlaying = true;
                    this._ifpStartRaf();
                });
                el.addEventListener("pause", () => {
                    if (el !== this._ifpActiveAudio || this._ifpSwapping) return;
                    this._ifpRealPlaying = false;
                    this._ifpStopRaf();
                    this._ifpDraw();
                });
                el.addEventListener("ended", () => {
                    if (el !== this._ifpActiveAudio || this._ifpSwapping) return;
                    this._ifpRealPlaying = false;
                    this._ifpStopRaf();
                    this._ifpDraw();
                });
            };
            setupAudioEvents(audioA);
            setupAudioEvents(audioB);
        };

        // --- Loop bar helpers ---

        nodeType.prototype._ifpGetLayout = function () {
            const canvas = this._ifpCanvas;
            if (!canvas) return null;
            const rect = canvas.getBoundingClientRect();
            const w = rect.width;
            const h = rect.height;
            if (w === 0 || h === 0) return null;
            const marginLeft = 60;
            const marginRight = 20;
            const marginTop = 36;
            const graphX = marginLeft;
            const graphW = w - marginLeft - marginRight;
            const loopBarY = marginTop - LOOP_BAR_H - 2;
            return { w, h, graphX, graphW, marginTop, loopBarY };
        };

        nodeType.prototype._ifpCanvasPos = function (e) {
            const rect = this._ifpCanvas.getBoundingClientRect();
            return { x: e.clientX - rect.left, y: e.clientY - rect.top };
        };

        nodeType.prototype._ifpXToFrac = function (x, layout) {
            return Math.max(0, Math.min(1, (x - layout.graphX) / layout.graphW));
        };

        nodeType.prototype._ifpOnMouseDown = function (e) {
            if (!this._ifpAudioShown) return;
            const layout = this._ifpGetLayout();
            if (!layout) return;
            const pos = this._ifpCanvasPos(e);
            const loop = this._ifpLoop;

            if (pos.y < layout.loopBarY || pos.y > layout.loopBarY + LOOP_BAR_H) return;

            const frac = this._ifpXToFrac(pos.x, layout);
            const startX = layout.graphX + loop.start * layout.graphW;
            const endX = layout.graphX + loop.end * layout.graphW;

            if (loop.enabled) {
                if (Math.abs(pos.x - startX) <= HANDLE_W + 2) {
                    this._ifpDrag = { type: "start", originStart: loop.start, originEnd: loop.end };
                    this._ifpCanvas.style.cursor = "ew-resize";
                    e.preventDefault();
                    return;
                }
                if (Math.abs(pos.x - endX) <= HANDLE_W + 2) {
                    this._ifpDrag = { type: "end", originStart: loop.start, originEnd: loop.end };
                    this._ifpCanvas.style.cursor = "ew-resize";
                    e.preventDefault();
                    return;
                }
                if (frac >= loop.start && frac <= loop.end) {
                    this._ifpDrag = { type: "move", originX: frac, originStart: loop.start, originEnd: loop.end };
                    this._ifpCanvas.style.cursor = "grab";
                    e.preventDefault();
                    return;
                }
            }

            this._ifpDrag = { type: "create", originX: frac };
            loop.start = frac;
            loop.end = frac;
            loop.enabled = true;
            this._ifpCanvas.style.cursor = "ew-resize";
            e.preventDefault();
        };

        nodeType.prototype._ifpOnMouseMove = function (e) {
            const drag = this._ifpDrag;
            if (!drag) {
                if (this._ifpAudioShown && this._ifpLoop.enabled) {
                    const layout = this._ifpGetLayout();
                    if (!layout) return;
                    const pos = this._ifpCanvasPos(e);
                    if (pos.y >= layout.loopBarY && pos.y <= layout.loopBarY + LOOP_BAR_H) {
                        const loop = this._ifpLoop;
                        const startX = layout.graphX + loop.start * layout.graphW;
                        const endX = layout.graphX + loop.end * layout.graphW;
                        if (Math.abs(pos.x - startX) <= HANDLE_W + 2 || Math.abs(pos.x - endX) <= HANDLE_W + 2) {
                            this._ifpCanvas.style.cursor = "ew-resize";
                        } else {
                            const frac = this._ifpXToFrac(pos.x, layout);
                            this._ifpCanvas.style.cursor = (frac >= loop.start && frac <= loop.end) ? "grab" : "crosshair";
                        }
                    } else {
                        this._ifpCanvas.style.cursor = "default";
                    }
                }
                return;
            }

            const layout = this._ifpGetLayout();
            if (!layout) return;
            const pos = this._ifpCanvasPos(e);
            const frac = this._ifpXToFrac(pos.x, layout);
            const loop = this._ifpLoop;

            if (drag.type === "create") {
                loop.start = Math.min(drag.originX, frac);
                loop.end = Math.max(drag.originX, frac);
            } else if (drag.type === "start") {
                loop.start = Math.min(frac, loop.end - 0.005);
            } else if (drag.type === "end") {
                loop.end = Math.max(frac, loop.start + 0.005);
            } else if (drag.type === "move") {
                const delta = frac - drag.originX;
                let newStart = drag.originStart + delta;
                let newEnd = drag.originEnd + delta;
                const span = newEnd - newStart;
                if (newStart < 0) { newStart = 0; newEnd = span; }
                if (newEnd > 1) { newEnd = 1; newStart = 1 - span; }
                loop.start = newStart;
                loop.end = newEnd;
            }

            this._ifpDraw();
            e.preventDefault();
        };

        nodeType.prototype._ifpOnMouseUp = function () {
            if (this._ifpDrag) {
                const loop = this._ifpLoop;
                if (loop.end - loop.start < 0.01) {
                    loop.enabled = false;
                    loop.start = 0;
                    loop.end = 1;
                }
                this._ifpDrag = null;
                this._ifpCanvas.style.cursor = "default";
                this._ifpDraw();
            }
        };

        nodeType.prototype._ifpOnDblClick = function (e) {
            if (!this._ifpAudioShown) return;
            const layout = this._ifpGetLayout();
            if (!layout) return;
            const pos = this._ifpCanvasPos(e);
            if (pos.y >= layout.loopBarY && pos.y <= layout.loopBarY + LOOP_BAR_H) {
                this._ifpLoop.enabled = false;
                this._ifpLoop.start = 0;
                this._ifpLoop.end = 1;
                this._ifpDraw();
                e.preventDefault();
            }
        };

        nodeType.prototype._ifpCheckLoop = function () {
            if (this._ifpSwapping) return;
            const loop = this._ifpLoop;
            if (!loop.enabled || !this._ifpAudioDuration) return;
            const audioEl = this._ifpActiveAudio;
            const endTime = loop.end * this._ifpAudioDuration;
            const startTime = loop.start * this._ifpAudioDuration;
            if (audioEl.currentTime >= endTime) {
                audioEl.currentTime = startTime;
            }
        };

        // --- Audio (double-buffered for seamless swaps) ---

        nodeType.prototype._ifpHandleAudio = function (audioFile, duration, rewindSeconds) {
            this._ifpAudioDuration = duration || 0;

            if (!this._ifpAudioShown) {
                this._ifpAudioShown = true;
                this._ifpAudioWrapper.style.display = "block";
                this._ifpSized = false;
            }

            const prevTime = this._ifpRealTime;
            const wasPlaying = this._ifpRealPlaying;
            const newUrl = buildAudioUrl(audioFile);

            // Cancel any in-flight swap so stale callbacks can't fire
            if (this._ifpSwapAbort) {
                this._ifpSwapAbort.abort();
            }
            const abort = new AbortController();
            this._ifpSwapAbort = abort;

            this._ifpSwapping = true;

            const current = this._ifpActiveAudio;
            const incoming = this._ifpInactiveAudio;

            // Load new audio into the inactive (hidden) element while the
            // current element keeps playing uninterrupted.
            incoming.src = newUrl;
            incoming.addEventListener("canplay", () => {
                if (abort.signal.aborted) return;

                // Use the live playback position from the still-playing audio
                // so the rewind offset accounts for time elapsed during loading.
                const liveTime = !current.paused ? current.currentTime : prevTime;
                const targetTime = Math.max(0, liveTime - rewindSeconds);
                incoming.currentTime = targetTime;

                incoming.addEventListener("seeked", () => {
                    if (abort.signal.aborted) return;

                    this._ifpSwapping = false;
                    this._ifpSwapAbort = null;

                    // Start new audio before stopping old to minimize any gap
                    if (wasPlaying) {
                        incoming.play();
                    }
                    current.pause();

                    // Swap visibility
                    current.style.display = "none";
                    incoming.style.display = "block";

                    // Swap references
                    this._ifpActiveAudio = incoming;
                    this._ifpInactiveAudio = current;

                    this._ifpRealTime = incoming.currentTime;
                }, { once: true, signal: abort.signal });
            }, { once: true, signal: abort.signal });

            incoming.addEventListener("error", () => {
                if (abort.signal.aborted) return;
                this._ifpSwapping = false;
                this._ifpSwapAbort = null;
            }, { once: true, signal: abort.signal });
        };

        // --- RAF ---

        nodeType.prototype._ifpStartRaf = function () {
            if (this._ifpRafId) return;
            const loop = () => {
                this._ifpCheckLoop();
                this._ifpDraw();
                this._ifpRafId = requestAnimationFrame(loop);
            };
            this._ifpRafId = requestAnimationFrame(loop);
        };

        nodeType.prototype._ifpStopRaf = function () {
            if (this._ifpRafId) {
                cancelAnimationFrame(this._ifpRafId);
                this._ifpRafId = null;
            }
        };

        // --- Draw ---

        nodeType.prototype._ifpDraw = function () {
            const canvas = this._ifpCanvas;
            const data = this._ifpData;
            if (!canvas || !data) return;

            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            const w = rect.width;
            const h = rect.height;

            // Skip draw if canvas has no layout (e.g. node being dragged).
            // Don't touch canvas.width/height so the last good frame is preserved.
            if (w < 10 || h < 10) return;

            const bw = Math.round(w * dpr);
            const bh = Math.round(h * dpr);
            if (canvas.width !== bw || canvas.height !== bh) {
                canvas.width = bw;
                canvas.height = bh;
            }

            const ctx = canvas.getContext("2d");
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            const { series, title } = data;

            // Layout
            const marginLeft = 60;
            const marginRight = 20;
            const marginTop = 36;
            const legendRowHeight = 22;
            const legendRows = Math.ceil(series.length / 3);
            const marginBottom = 20 + legendRows * legendRowHeight;
            const graphX = marginLeft;
            const graphY = marginTop;
            const graphW = w - marginLeft - marginRight;
            const graphH = h - marginTop - marginBottom;

            if (graphW <= 0 || graphH <= 0) return;

            // Background gradient
            const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
            bgGrad.addColorStop(0, "#191925");
            bgGrad.addColorStop(1, "#1a1a2a");
            ctx.fillStyle = bgGrad;
            ctx.fillRect(0, 0, w, h);

            // --- Loop bar (above graph, below title) ---
            if (this._ifpAudioShown) {
                const loopBarY = marginTop - LOOP_BAR_H - 2;
                ctx.fillStyle = "#252535";
                ctx.fillRect(graphX, loopBarY, graphW, LOOP_BAR_H);
                ctx.strokeStyle = "#3a3a4a";
                ctx.lineWidth = 1;
                ctx.strokeRect(graphX, loopBarY, graphW, LOOP_BAR_H);

                ctx.strokeStyle = "#3a3a4a";
                const tickCount = 20;
                for (let i = 1; i < tickCount; i++) {
                    const tx = graphX + (i / tickCount) * graphW;
                    const tickH = i % 5 === 0 ? LOOP_BAR_H * 0.5 : LOOP_BAR_H * 0.25;
                    ctx.beginPath();
                    ctx.moveTo(tx, loopBarY + LOOP_BAR_H);
                    ctx.lineTo(tx, loopBarY + LOOP_BAR_H - tickH);
                    ctx.stroke();
                }

                const loop = this._ifpLoop;
                if (loop.enabled) {
                    const lx = graphX + loop.start * graphW;
                    const lw = (loop.end - loop.start) * graphW;

                    ctx.fillStyle = "rgba(80, 180, 255, 0.25)";
                    ctx.fillRect(lx, loopBarY, lw, LOOP_BAR_H);
                    ctx.strokeStyle = "rgba(80, 180, 255, 0.7)";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(lx, loopBarY, lw, LOOP_BAR_H);

                    ctx.fillStyle = "rgba(80, 180, 255, 0.9)";
                    ctx.fillRect(lx - HANDLE_W / 2, loopBarY, HANDLE_W, LOOP_BAR_H);
                    ctx.fillRect(lx + lw - HANDLE_W / 2, loopBarY, HANDLE_W, LOOP_BAR_H);

                    ctx.fillStyle = "rgba(80, 180, 255, 0.06)";
                    ctx.fillRect(lx, graphY, lw, graphH);
                    ctx.strokeStyle = "rgba(80, 180, 255, 0.3)";
                    ctx.setLineDash([4, 4]);
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(lx, graphY);
                    ctx.lineTo(lx, graphY + graphH);
                    ctx.moveTo(lx + lw, graphY);
                    ctx.lineTo(lx + lw, graphY + graphH);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }

            // Y range
            let yMin = Infinity, yMax = -Infinity;
            for (const s of series) {
                for (const v of s.values) {
                    if (v < yMin) yMin = v;
                    if (v > yMax) yMax = v;
                }
            }
            if (yMin === yMax) yMax = yMin + 1;
            const yRange = yMax - yMin;
            const pad = 0.05 * yRange;
            yMin -= pad;
            yMax += pad;

            // Grid
            ctx.lineWidth = 1;
            for (let i = 0; i <= 8; i++) {
                const y = graphY + (i * graphH / 8);
                ctx.strokeStyle = i % 2 === 0 ? "#41414b" : "#2d2d37";
                ctx.beginPath();
                ctx.moveTo(graphX, y);
                ctx.lineTo(graphX + graphW, y);
                ctx.stroke();
            }
            for (let i = 0; i <= 12; i++) {
                const x = graphX + (i * graphW / 12);
                ctx.strokeStyle = i % 3 === 0 ? "#41414b" : "#2d2d37";
                ctx.beginPath();
                ctx.moveTo(x, graphY);
                ctx.lineTo(x, graphY + graphH);
                ctx.stroke();
            }

            // Border
            ctx.strokeStyle = "#787890";
            ctx.lineWidth = 2;
            ctx.strokeRect(graphX - 1, graphY - 1, graphW + 2, graphH + 2);
            ctx.strokeStyle = "#c8c8dc";
            ctx.lineWidth = 1;
            ctx.strokeRect(graphX, graphY, graphW, graphH);

            // Y-axis labels
            ctx.fillStyle = "#8c8ca0";
            ctx.font = "11px monospace";
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            for (let i = 0; i <= 4; i++) {
                const yVal = yMin + (yMax - yMin) * (1 - i / 4);
                const yPos = graphY + (i * graphH / 4);
                ctx.fillText(yVal.toFixed(2), graphX - 8, yPos);
            }

            // X-axis frame labels
            const frameCount = series[0]?.values.length || 1;
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            const xSteps = Math.min(12, frameCount - 1);
            if (xSteps > 0) {
                for (let i = 0; i <= xSteps; i++) {
                    const frame = Math.round(i * (frameCount - 1) / xSteps);
                    const x = graphX + (frame / Math.max(frameCount - 1, 1)) * graphW;
                    ctx.fillStyle = "#8c8ca0";
                    ctx.fillText(frame.toString(), x, graphY + graphH + 4);
                }
            }

            // Draw series lines
            ctx.save();
            ctx.beginPath();
            ctx.rect(graphX, graphY, graphW, graphH);
            ctx.clip();

            for (const s of series) {
                const [r, g, b] = s.color;
                const values = s.values;
                const len = values.length;
                if (len < 1) continue;

                const points = new Array(len);
                for (let i = 0; i < len; i++) {
                    const x = graphX + (i / Math.max(len - 1, 1)) * graphW;
                    const yRatio = (values[i] - yMin) / (yMax - yMin);
                    const y = graphY + graphH - yRatio * graphH;
                    points[i] = [x, y];
                }

                // Glow
                ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.25)`;
                ctx.lineWidth = 5;
                ctx.lineJoin = "round";
                ctx.lineCap = "round";
                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                for (let i = 1; i < len; i++) ctx.lineTo(points[i][0], points[i][1]);
                ctx.stroke();

                // Main line
                ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                for (let i = 1; i < len; i++) ctx.lineTo(points[i][0], points[i][1]);
                ctx.stroke();
            }

            ctx.restore();

            // Playhead
            if (this._ifpAudioShown && this._ifpAudioDuration > 0) {
                const currentTime = this._ifpSwapping ? this._ifpRealTime : this._ifpActiveAudio.currentTime;
                const progress = currentTime / this._ifpAudioDuration;
                if (progress >= 0 && progress <= 1) {
                    const px = graphX + progress * graphW;

                    ctx.strokeStyle = "rgba(255, 60, 60, 0.8)";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(px, graphY);
                    ctx.lineTo(px, graphY + graphH);
                    ctx.stroke();

                    // Triangle marker
                    ctx.fillStyle = "rgba(255, 60, 60, 0.9)";
                    ctx.beginPath();
                    ctx.moveTo(px, graphY);
                    ctx.lineTo(px - 5, graphY - 8);
                    ctx.lineTo(px + 5, graphY - 8);
                    ctx.closePath();
                    ctx.fill();

                    // Playhead on loop bar
                    const loopBarY = marginTop - LOOP_BAR_H - 2;
                    ctx.fillStyle = "rgba(255, 60, 60, 0.9)";
                    ctx.fillRect(px - 1, loopBarY, 2, LOOP_BAR_H);
                }
            }

            // Title bar
            ctx.font = "bold 15px sans-serif";
            const titleW = ctx.measureText(title).width;
            ctx.fillStyle = "rgba(20, 20, 40, 0.7)";
            ctx.fillRect((w - titleW) / 2 - 12, 4, titleW + 24, 24);
            ctx.fillStyle = "#ffffff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(title, w / 2, 16);

            // Legend
            const legendY = graphY + graphH + 22;
            const cols = 3;
            const colWidth = graphW / cols;
            ctx.font = "12px monospace";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            for (let i = 0; i < series.length; i++) {
                const s = series[i];
                const [r, g, b] = s.color;
                const row = Math.floor(i / cols);
                const col = i % cols;
                const lx = graphX + col * colWidth;
                const ly = legendY + row * legendRowHeight;

                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(lx, ly - 6, 14, 12);
                ctx.strokeStyle = "#c8c8c8";
                ctx.lineWidth = 1;
                ctx.strokeRect(lx, ly - 6, 14, 12);

                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillText(s.label, lx + 20, ly);
            }
        };

        // Clean up on node removal
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._ifpStopRaf?.();
            if (this._ifpResizeObserver) {
                this._ifpResizeObserver.disconnect();
                this._ifpResizeObserver = null;
            }
            if (this._ifpAudioA) {
                this._ifpAudioA.pause();
                this._ifpAudioA.src = "";
            }
            if (this._ifpAudioB) {
                this._ifpAudioB.pause();
                this._ifpAudioB.src = "";
            }
            onRemoved?.apply(this, arguments);
        };
    },
});
