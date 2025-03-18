import * as THREE from './three.module.js';
import { OrbitControls } from './OrbitControls.js';
import * as dat from './lil-gui.module.min.js';
import { api } from './api.js';

// 获取 DOM 元素
const visualizer = document.getElementById("visualizer");
const container = document.getElementById("container");
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");

// 设置 GUI 控件
const gui = new dat.GUI({ width: 150 });
const initialFov = 75;
const params = {
    fov: initialFov
};

// 添加 FOV 控件
gui.add(params, 'fov', 10, 120).name('缩放').step(0.1).onChange(() => {
    camera.fov = params.fov;
    camera.updateProjectionMatrix();
});
gui.hide();

// 初始化渲染器
const renderer = new THREE.WebGLRenderer({
    antialias: true,
    extensions: { derivatives: true }
});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

// 创建 PMREM 生成器
const pmremGenerator = new THREE.PMREMGenerator(renderer);

// 创建场景
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// 添加环境光
const ambientLight = new THREE.AmbientLight(0xffffff);
scene.add(ambientLight);

// 设置相机
const camera = new THREE.PerspectiveCamera(initialFov, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 5);

// 添加点光源
const pointLight = new THREE.PointLight(0xffffff, 15);
camera.add(pointLight);
scene.add(camera);

// 初始化轨道控制器
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enablePan = true;
controls.enableDamping = true;

// 处理窗口大小调整
window.addEventListener('resize', onWindowResize, false);

//显示控制
gui.show();

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}


// 状态变量
let lastpanoImage = "";
let needUpdate = false;

// 帧更新函数
function frameUpdate() {
    const panoImage = visualizer.getAttribute("pano_image");

    if (panoImage === lastpanoImage) {
        if (needUpdate) {
            controls.update();
            renderer.render(scene, camera);
        }
        requestAnimationFrame(frameUpdate);
    } else {
        needUpdate = false;
        scene.clear();
        progressDialog.open = true;
        lastpanoImage = panoImage;
        main(JSON.parse(lastpanoImage));
    }
}

// 主函数：加载 pano 图像并设置场景
async function main(panoImageParams) {
    let panoTexture;
    console.log("加载 pano 参数:", panoImageParams);

    if (panoImageParams?.filename) {
        const panoImageUrl = api.apiURL('/view?' + new URLSearchParams(panoImageParams)).replace(/extensions.*\//, "");
        const panoLoader = new THREE.TextureLoader();
        try {
            panoTexture = await panoLoader.loadAsync(panoImageUrl);
            console.log("加载 pano 纹理:", panoTexture);
            panoTexture.mapping = THREE.EquirectangularReflectionMapping;
            panoTexture.colorSpace = THREE.SRGBColorSpace;
        } catch (error) {
            console.error("加载 pano 图像失败:", error);
        }
    }

    if (panoTexture) {
        scene.environment = panoTexture;
        scene.background = panoTexture;
    }

    needUpdate = true;
    progressDialog.close();

    frameUpdate();
}

// 初始化调用
main();

