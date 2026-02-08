import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "RyanOnTheInside.DisplayAny",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ROTIDisplayAny") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (!this.displayWidget) {
                const w = ComfyWidgets["STRING"](this, "value", ["STRING", { multiline: true }], app);
                this.displayWidget = w.widget;
                this.displayWidget.inputEl.readOnly = true;
                this.displayWidget.inputEl.style.opacity = "0.85";
                this.displayWidget.inputEl.style.fontFamily = "monospace";
                this.displayWidget.inputEl.style.fontSize = "12px";
                this.displayWidget.serialize = false;
            }

            if (message?.text?.[0] !== undefined) {
                this.displayWidget.value = message.text[0];
            }

            requestAnimationFrame(() => {
                const lines = (this.displayWidget.value?.split("\n").length || 1);
                const minHeight = Math.min(400, Math.max(120, lines * 16 + 80));
                if (this.size[1] < minHeight) {
                    this.setSize([Math.max(this.size[0], 300), minHeight]);
                }
            });
        };
    },
});
