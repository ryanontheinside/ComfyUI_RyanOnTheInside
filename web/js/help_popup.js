import { app } from "../../../scripts/app.js";

// code based on mtb nodes by Mel Massadian https://github.com/melMass/comfy_mtb/
export const loadScript = (
  FILE_URL,
  async = true,
  type = 'text/javascript',
) => {
  return new Promise((resolve, reject) => {
    try {
      // Check if the script already exists
      const existingScript = document.querySelector(`script[src="${FILE_URL}"]`)
      if (existingScript) {
        resolve({ status: true, message: 'Script already loaded' })
        return
      }

      const scriptEle = document.createElement('script')
      scriptEle.type = type
      scriptEle.async = async
      scriptEle.src = FILE_URL

      scriptEle.addEventListener('load', (ev) => {
        resolve({ status: true })
      })

      scriptEle.addEventListener('error', (ev) => {
        reject({
          status: false,
          message: `Failed to load the script ${FILE_URL}`,
        })
      })

      document.body.appendChild(scriptEle)
    } catch (error) {
      reject(error)
    }
  })
}

loadScript('/ryanontheinside_web_async/marked.min.js').catch((e) => {
  console.log(e)
})
loadScript('/ryanontheinside_web_async/purify.min.js').catch((e) => {
  console.log(e)
})

const categories = ["RyanOnTheInside"];
app.registerExtension({
	name: "RyanOnTheInside.HelpPopup",
	async beforeRegisterNodeDef(nodeType, nodeData) {
   
  if (app.ui.settings.getSettingValue("RyanOnTheInside.helpPopup") === false) {
    return;
    }
		try {
			categories.forEach(category => {
        if (nodeData?.category?.startsWith(category)) {
            addDocumentation(nodeData, nodeType);
        }
        else return
    });
		} catch (error) {
			console.error("Error in registering RyanOnTheInside.HelpPopup", error);
		}
	},
});

const create_documentation_stylesheet = () => {
    const tag = 'roti-documentation-stylesheet'

    let styleTag = document.head.querySelector(tag)

    if (!styleTag) {
      styleTag = document.createElement('style')
      styleTag.type = 'text/css'
      styleTag.id = tag
      styleTag.innerHTML = `
      .roti-documentation-popup {
        background: var(--comfy-menu-bg);
        position: absolute;
        color: var(--fg-color);
        font: 12px monospace;
        line-height: 1.5em;
        padding: 10px;
        border-radius: 10px;
        border-style: solid;
        border-width: medium;
        border-color: var(--border-color);
        z-index: 5;
        overflow: hidden;
        opacity: 0;
        transform: scale(0.95);
        animation: popup-appear 0.3s ease forwards, popup-pulse 2s ease-in-out infinite;
       }

       @keyframes popup-appear {
         from {
           opacity: 0;
           transform: scale(0.95);
         }
         to {
           opacity: 1;
           transform: scale(1);
         }
       }

       @keyframes popup-pulse {
         0% {
           box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.4);
         }
         50% {
           box-shadow: 0 0 20px 10px rgba(255, 165, 0, 0.2);
         }
         100% {
           box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.4);
         }
       }

       .roti-documentation-popup.closing {
         animation: popup-close 0.2s ease forwards;
       }

       @keyframes popup-close {
         from {
           opacity: 1;
           transform: scale(1);
         }
         to {
           opacity: 0;
           transform: scale(0.95) translateY(-10px);
         }
       }

       .content-wrapper {
        overflow: auto;
        max-height: 100%;
        /* Scrollbar styling for Chrome */
        &::-webkit-scrollbar {
           width: 6px;
        }
        &::-webkit-scrollbar-track {
           background: var(--bg-color);
        }
        &::-webkit-scrollbar-thumb {
           background-color: var(--fg-color);
           border-radius: 6px;
           border: 3px solid var(--bg-color);
        }
       
        /* Scrollbar styling for Firefox */
        scrollbar-width: thin;
        scrollbar-color: var(--fg-color) var(--bg-color);
        a {
          color: yellow;
        }
        a:visited {
          color: orange;
        }
        a:hover {
          color: red;
        }
       }
        `
      document.head.appendChild(styleTag)
    }
  } 

  /** Add documentation widget to the selected node */
  export const addDocumentation = (
    nodeData,
    nodeType,
    opts = { icon_size: 14, icon_margin: 4 },) => {

    opts = opts || {}
    const iconSize = opts.icon_size ? opts.icon_size : 14
    const iconMargin = opts.icon_margin ? opts.icon_margin : 4

    // Store popup elements in the node instance instead of function scope
    nodeType.prototype._docElement = null;
    nodeType.prototype._contentWrapper = null;
    nodeType.prototype._docCtrl = null;

    const drawFg = nodeType.prototype.onDrawForeground
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = drawFg ? drawFg.apply(this, arguments) : undefined
      if (this.flags.collapsed) return r

      // icon position
      const x = this.size[0] - iconSize - iconMargin
      
      // create the popup
      if (this.show_doc && !this._docElement) {
        // Clean up any existing elements first
        this.cleanupDocumentation();
        
        this._docElement = document.createElement('div')
        this._contentWrapper = document.createElement('div');
        this._docElement.appendChild(this._contentWrapper);

        create_documentation_stylesheet()
        this._contentWrapper.classList.add('content-wrapper');
        this._docElement.classList.add('roti-documentation-popup')
        
        // Construct the content with default links and optional help text
        let content = "";
        
        // Add ASCII art banner
        content += `
\`\`\`
██████╗  ██╗   ██╗ █████╗ ███╗   ██╗ ██████╗ ███╗   ██╗████████╗██╗  ██╗███████╗██╗███╗   ██╗███████╗██╗██████╗ ███████╗
██╔══██╗ ╚██╗ ██╔╝██╔══██╗████╗  ██║██╔═══██╗████╗  ██║╚══██╔══╝██║  ██║██╔════╝██║████╗  ██║██╔════╝██║██╔══██╗██╔════╝
██████╔╝  ╚████╔╝ ███████║██╔██╗ ██║██║   ██║██╔██╗ ██║   ██║   ███████║█████╗  ██║██╔██╗ ██║███████╗██║██║  ██║█████╗  
██╔══██╗   ╚██╔╝  ██╔══██║██║╚██╗██║██║   ██║██║╚██╗██║   ██║   ██╔══██║██╔══╝  ██║██║╚██╗██║╚════██║██║██║  ██║██╔══╝  
██║  ██║    ██║   ██║  ██║██║ ╚████║╚██████╔╝██║ ╚████║   ██║   ██║  ██║███████╗██║██║ ╚████║███████║██║██████╔╝███████╗
╚═╝  ╚═╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝╚═════╝ ╚══════╝
\`\`\`

`;
        
        // Add node-specific help text if available
        if (nodeData.help_text) {
            content += nodeData.help_text + "\n\n---\n\n";
        }
        
        // Add default links footer
        content += `
## For more information, visit [RyanOnTheInside GitHub](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside).

## For tutorials and example workflows visit [RyanOnTheInside Civitai](https://civitai.com/user/ryanontheinside).

## For video tutorials and more visit [RyanOnTheInside YouTube](https://www.youtube.com/@ryanontheinside).

## [RyanOnTheInside Linktree](https://linktr.ee/ryanontheinside)
`;
        
        //parse the combined content with marked and sanitize
        this._contentWrapper.innerHTML = DOMPurify.sanitize(marked.parse(content))

        // resize handle
        const resizeHandle = document.createElement('div');
        resizeHandle.style.width = '0';
        resizeHandle.style.height = '0';
        resizeHandle.style.position = 'absolute';
        resizeHandle.style.bottom = '0';
        resizeHandle.style.right = '0';
        resizeHandle.style.cursor = 'se-resize';
        
        // Add pseudo-elements to create a triangle shape
        const borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim();
        resizeHandle.style.borderTop = '10px solid transparent';
        resizeHandle.style.borderLeft = '10px solid transparent';
        resizeHandle.style.borderBottom = `10px solid ${borderColor}`;
        resizeHandle.style.borderRight = `10px solid ${borderColor}`;

        this._docElement.appendChild(resizeHandle)
        let isResizing = false
        let startX, startY, startWidth, startHeight

        // Create new AbortController for this instance
        this._docCtrl = new AbortController();

        resizeHandle.addEventListener('mousedown', (e) => {
          e.preventDefault();
          e.stopPropagation();
          isResizing = true;
          startX = e.clientX;
          startY = e.clientY;
          startWidth = parseInt(document.defaultView.getComputedStyle(this._docElement).width, 10);
          startHeight = parseInt(document.defaultView.getComputedStyle(this._docElement).height, 10);
        }, { signal: this._docCtrl.signal });

        // close button
        const closeButton = document.createElement('div');
        closeButton.textContent = '❌';
        closeButton.style.position = 'absolute';
        closeButton.style.top = '0';
        closeButton.style.right = '0';
        closeButton.style.cursor = 'pointer';
        closeButton.style.padding = '5px';
        closeButton.style.color = 'red';
        closeButton.style.fontSize = '12px';

        this._docElement.appendChild(closeButton)

        closeButton.addEventListener('mousedown', (e) => {
          e.stopPropagation();
          this.show_doc = false;
          this.cleanupDocumentation();
        }, { signal: this._docCtrl.signal });

        document.addEventListener('mousemove', (e) => {
          if (!isResizing) return;
          const scale = app.canvas.ds.scale;
          const newWidth = startWidth + (e.clientX - startX) / scale;
          const newHeight = startHeight + (e.clientY - startY) / scale;
          if (this._docElement) {
            this._docElement.style.width = `${newWidth}px`;
            this._docElement.style.height = `${newHeight}px`;
          }
        }, { signal: this._docCtrl.signal });

        document.addEventListener('mouseup', () => {
          isResizing = false;
        }, { signal: this._docCtrl.signal });

        document.body.appendChild(this._docElement)
      }
      // close the popup
      else if (!this.show_doc && this._docElement) {
        this.cleanupDocumentation();
      }

      // update position of the popup
      if (this.show_doc && this._docElement) {
        const rect = ctx.canvas.getBoundingClientRect()
        const scaleX = rect.width / ctx.canvas.width
        const scaleY = rect.height / ctx.canvas.height

        const transform = new DOMMatrix()
        .scaleSelf(scaleX, scaleY)
        .multiplySelf(ctx.getTransform())
        .translateSelf(this.size[0] * scaleX * Math.max(1.0,window.devicePixelRatio) , 0)
        .translateSelf(10, -32)
        
        const scale = new DOMMatrix()
        .scaleSelf(transform.a, transform.d);

        const styleObject = {
          transformOrigin: '0 0',
          transform: scale,
          left: `${transform.a + transform.e}px`,
          top: `${transform.d + transform.f}px`,
        };
        Object.assign(this._docElement.style, styleObject);
      }

      ctx.save()
      ctx.translate(x - 2, iconSize - 34)
      ctx.scale(iconSize / 32, iconSize / 32)
      ctx.strokeStyle = 'rgba(255,255,255,0.3)'
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.lineWidth = 2.4
      ctx.font = 'bold 36px monospace'
      ctx.fillStyle = 'orange';
      ctx.fillText('?', 0, 24)
      ctx.restore()
      return r
    }

    // Add cleanup method to the prototype
    nodeType.prototype.cleanupDocumentation = function() {
      if (this._docCtrl) {
        this._docCtrl.abort();
        this._docCtrl = null;
      }
      
      if (this._docElement) {
        this._docElement.classList.add('closing');
        const cleanup = () => {
          if (this._docElement && this._docElement.parentNode) {
            this._docElement.parentNode.removeChild(this._docElement);
          }
          if (this._contentWrapper) {
            this._contentWrapper.remove();
          }
          this._docElement = null;
          this._contentWrapper = null;
        };
        
        this._docElement.addEventListener('animationend', cleanup, { once: true });
      }
    }

    // handle clicking of the icon
    const mouseDown = nodeType.prototype.onMouseDown
    nodeType.prototype.onMouseDown = function (e, localPos, canvas) {
      const r = mouseDown ? mouseDown.apply(this, arguments) : undefined
      const iconX = this.size[0] - iconSize - iconMargin
      const iconY = iconSize - 34
      if (
        localPos[0] > iconX &&
        localPos[0] < iconX + iconSize &&
        localPos[1] > iconY &&
        localPos[1] < iconY + iconSize
      ) {
        // Clean up existing popup before toggling
        if (this.show_doc) {
          this.cleanupDocumentation();
        }
        this.show_doc = !this.show_doc;
        return true;
      }
      return r;
    }

    const onRem = nodeType.prototype.onRemoved
    nodeType.prototype.onRemoved = function () {
      const r = onRem ? onRem.apply(this, []) : undefined
      this.cleanupDocumentation();
      return r;
    }
}