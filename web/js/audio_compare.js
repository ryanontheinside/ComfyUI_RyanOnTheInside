import { app } from "../../../scripts/app.js";

function buildAudioUrl(file) {
    const params = new URLSearchParams({
        filename: file.filename,
        type: file.type,
        subfolder: file.subfolder || "",
    });
    return `/view?${params.toString()}`;
}

app.registerExtension({
    name: "RyanOnTheInside.AudioCompare",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PreviewAudioCompare") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const aFiles = message?.a_audio;
            const bFiles = message?.b_audio;
            if (!aFiles?.length || !bFiles?.length) return;

            const urlA = buildAudioUrl(aFiles[0]);
            const urlB = buildAudioUrl(bFiles[0]);

            // Create the widget container once
            if (!this._abContainer) {
                const container = document.createElement("div");
                container.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:8px;";

                // Toggle button
                const btn = document.createElement("button");
                btn.style.cssText =
                    "padding:6px 16px;font-weight:bold;font-size:14px;cursor:pointer;" +
                    "border:2px solid #666;border-radius:6px;background:#333;color:#fff;" +
                    "align-self:center;min-width:80px;user-select:none;";
                btn.textContent = "A";
                container.appendChild(btn);

                // Audio element
                const audio = document.createElement("audio");
                audio.controls = true;
                audio.style.cssText = "width:100%;";
                container.appendChild(audio);

                // Label
                const label = document.createElement("div");
                label.style.cssText = "text-align:center;font-size:11px;color:#aaa;";
                label.textContent = "Playing: A";
                container.appendChild(label);

                // State
                this._abState = "A";
                this._abAudioEl = audio;
                this._abBtn = btn;
                this._abLabel = label;
                this._abContainer = container;

                btn.addEventListener("click", () => {
                    const currentTime = this._abAudioEl.currentTime;
                    const wasPlaying = !this._abAudioEl.paused;

                    if (this._abState === "A") {
                        this._abState = "B";
                        this._abAudioEl.src = this._abUrlB;
                        btn.textContent = "B";
                        btn.style.borderColor = "#e06030";
                        label.textContent = "Playing: B";
                    } else {
                        this._abState = "A";
                        this._abAudioEl.src = this._abUrlA;
                        btn.textContent = "A";
                        btn.style.borderColor = "#3080e0";
                        label.textContent = "Playing: A";
                    }

                    this._abAudioEl.currentTime = currentTime;
                    if (wasPlaying) {
                        this._abAudioEl.play();
                    }
                });

                // Create a DOM widget
                const widget = this.addDOMWidget("audio_compare", "custom", container, {
                    serialize: false,
                    hideOnZoom: false,
                });
                widget.computeSize = () => [this.size[0], 100];
            }

            // Update URLs
            this._abUrlA = urlA;
            this._abUrlB = urlB;

            // Reset to A
            this._abState = "A";
            this._abBtn.textContent = "A";
            this._abBtn.style.borderColor = "#3080e0";
            this._abLabel.textContent = "Playing: A";
            this._abAudioEl.src = urlA;
        };
    },
});
