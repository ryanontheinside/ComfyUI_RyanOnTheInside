import { app } from '../../../scripts/app.js'

class DoomWidget {
    constructor(node) {
        this.node = node;
        this.element = document.createElement('div');
        this.element.style.backgroundColor = "#121212";
        this.element.style.color = "white";
        
        // Create attribution link
        this.attributionLink = document.createElement("p");
        this.attributionLink.innerHTML = `<a href="https://linktr.ee/ryanontheinside" target="_blank" style="text-decoration: none; color: #007bff">Follow <span style="color: #007bff">RyanOnTheInside</span> For Even More Useful Stuff 🎮</a>`;
        this.attributionLink.style.textAlign = "center";
        this.attributionLink.style.marginBottom = "10px";
        
        // Create container for DOOM
        this.doomContainer = document.createElement("div");
        this.doomContainer.id = "DOOM";
        this.doomContainer.className = "dosbox-default";
        this.doomContainer.style.marginTop = "20px";
        
        // Add fullscreen button
        this.fullscreenBtn = document.createElement("p");
        this.fullscreenBtn.innerHTML = `<br><a id="fullscreen_DOOM" href="javascript: void 0;" class="fa-solid fa-expand"></a>&nbsp;&nbsp;FULLSCREEN`;
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            #DOOM > .dosbox-container { width: 640px; height: 400px; }
            #DOOM > .dosbox-container > .dosbox-overlay { background: url("https://thedoggybrad.github.io/doom_on_js-dos/DOOM.png"); }
            #DOOM > .dosbox-container > .dosbox-overlay > .dosbox-start {
                border-radius: 17px;
                background-color: rgba(90, 90, 90, 0.57);
                padding: 5px;
                text-align: center;
                width: 10em;
                margin: 0 auto;
            }
            a:link {color:white}
        `;
        
        // Append elements
        document.head.appendChild(style);
        this.element.appendChild(this.attributionLink);
        this.element.appendChild(this.doomContainer);
        this.element.appendChild(this.fullscreenBtn);
        
        // Load required scripts
        this.loadScripts().then(() => {
            this.initDoom();
        }).catch(error => {
            console.error("DoomWidget: Failed to load scripts:", error);
        });
    }

    async loadScripts() {
        try {
            await this.loadStylesheet("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css");
            await this.loadScript("https://code.jquery.com/jquery-3.7.1.min.js");
            await this.loadScript("https://thedoggybrad.github.io/doom_on_js-dos/js-dos-api.js");
        } catch (error) {
            console.error("DoomWidget: Error in loadScripts:", error);
            throw error;
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = () => {
                resolve();
            };
            script.onerror = (error) => {
                console.error("DoomWidget: Script failed to load:", src, error);
                reject(error);
            };
            document.head.appendChild(script);
        });
    }

    loadStylesheet(href) {
        return new Promise((resolve, reject) => {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = href;
            link.onload = () => {
                resolve();
            };
            link.onerror = (error) => {
                console.error("DoomWidget: Stylesheet failed to load:", href, error);
                reject(error);
            };
            document.head.appendChild(link);
        });
    }

    initDoom() {
        try {
            if (typeof Dosbox === 'undefined') {
                console.error("DoomWidget: Dosbox is not defined! js-dos API might not be loaded correctly");
                return;
            }

            const dosbox = new Dosbox({
                id: "DOOM",
                onload: (dosbox) => {
                    dosbox.run("https://thedoggybrad.github.io/doom_on_js-dos/DOOM-@evilution.zip", "./DOOM/DOOM.EXE");
                },
                onrun: (dosbox, app) => {
                }
            });

            const fullscreenBtn = document.getElementById("fullscreen_DOOM");
            if (fullscreenBtn) {
                fullscreenBtn.addEventListener("click", () => {
                    dosbox.requestFullScreen();
                });
            } else {
                console.error("DoomWidget: Fullscreen button not found in DOM");
            }
        } catch (error) {
            console.error("DoomWidget: Error in initDoom:", error);
        }
    }
}

app.registerExtension({
    name: 'Comfy.Doom',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        
        if (nodeData.name === 'Doom') {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply(this, arguments);
                
                // Create and add the DOOM widget
                const doomWidget = new DoomWidget(this);
                this.doom = this.addDOMWidget("doom", "DoomWidget", doomWidget.element, {
                    serialize: false,
                    hideOnZoom: false
                });
                
                // Set appropriate node size
                this.setSize([660, 508]);
                
                return r;
            }
        }
    }
});