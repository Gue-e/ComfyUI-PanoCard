import { app } from "../../scripts/app.js";

class Visualizer {
    constructor(node, container, visualSrc) {
        this.node = node;
        this.iframe = document.createElement('iframe');
        Object.assign(this.iframe, {
            scrolling: "no",
            style: {
                overflow: "hidden",
                border: "none",
                width: "100%",
                height: "100%",
                display: 'none' // 默认隐藏
            },
            src: `/extensions/ComfyUI-PanoCard/html/${visualSrc}.html`
        });
        container.appendChild(this.iframe);
    }

    updateVisual(params) {
        const iframeDocument = this.iframe.contentWindow.document;
        const previewScript = iframeDocument.getElementById('visualizer');
        previewScript.setAttribute("pano_image", JSON.stringify(params.pano_image));
    }

    remove() {
        this.iframe.remove();
    }

    show() {
        this.iframe.style.display = 'block';
    }

    hide() {
        this.iframe.style.display = 'none';
    }
}

function createVisualizer(node, inputName, typeName, app) {
    node.name = inputName;

    const widget = {
        type: typeName,
        name: "preview3d",
        callback: () => {},
        draw(ctx, node, widgetWidth, widgetY, widgetHeight) {
            const margin = 10;
            const topOffset = 35;
            const leftOffset = 40;
            const visible = app.canvas.ds.scale > 0.5 && this.type === typeName;
            const width = widgetWidth - margin * 4;
            const rect = ctx.canvas.getBoundingClientRect();

            const transform = new DOMMatrix()
                .scale(rect.width / ctx.canvas.width, rect.height / ctx.canvas.height)
                .multiply(ctx.getTransform())
                .translate(margin, margin + widgetY);

            Object.assign(this.visualizer.style, {
                left: `${transform.a * margin + transform.e + leftOffset}px`,
                top: `${transform.d + transform.f + topOffset}px`,
                width: `${width * transform.a}px`,
                height: `${width * transform.d - widgetHeight - (8 * margin) * transform.d}px`,
                position: "absolute",
                overflow: "hidden",
                zIndex: app.graph._nodes.indexOf(node),
                display: visible ? 'block' : 'none'
            });

            Object.assign(this.visualizer.firstChild.style, {
                transformOrigin: "50% 50%",
                width: '100%',
                height: '100%',
                border: 'none'
            });
        }
    };

    const container = document.createElement('div');
    container.id = `Comfy3D_${inputName}`;
    container.style.display = 'none'; // 初始状态为隐藏

    node.visualizer = new Visualizer(node, container, typeName);
    widget.visualizer = container;
    widget.parent = node;

    document.body.appendChild(container);

    node.addCustomWidget(widget);

    node.updateParameters = (params) => {
        node.visualizer.updateVisual(params);
    };

    node.onDrawBackground = function (ctx) {
        this.visualizer.iframe.hidden = this.flags.collapsed;
    };

    node.onResize = function () {
        let [w, h] = this.size;
        w = Math.max(w, 600);
        h = Math.max(h, 500);

        if (w > 600) {
            h = w - 100;
        }

        this.size = [w, h];
    };

    node.onRemoved = () => {
        Object.values(node.widgets).forEach(widget => {
            if (widget.visualizer) {
                widget.visualizer.remove();
            }
        });
    };

    return { widget };
}

function registerVisualizer(nodeType, nodeData, nodeClassName, typeName) {
    if (nodeData.name === nodeClassName) {
        console.log(`[3D Visualizer] Registering node: ${nodeData.name}`);

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = async function () {
            const result = originalOnNodeCreated
                ? await originalOnNodeCreated.apply(this, arguments)
                : undefined;

            const existingNodes = app.graph._nodes.filter(node => node.type === nodeClassName);
            const nodeName = `Preview3DNode_${existingNodes.length}`;

            console.log(`[Comfy3D] Create: ${nodeName}`);

            await createVisualizer(this, nodeName, typeName, app);

            this.setSize([600, 500]);

            return result;
        };

        nodeType.prototype.onExecuted = function (message) {
            console.log(message);
            if (message.pano_image) {
                const params = { pano_image: message.pano_image[0] };
                this.updateParameters(params);
                this.visualizer.show();
            } else {
                if (this.visualizer.parentNode) {
                    this.visualizer.hide();
                }
            }
        };
    }
}

app.registerExtension({
    name: "Gue.PanoCard.Viewer",

    async init(app) {
        // 初始化逻辑（如果需要）
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        registerVisualizer(nodeType, nodeData, "PanoCardViewer", "threeVisualizer");
    }
});