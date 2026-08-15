import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

/**
 * Real3DViewer
 *
 * 3D 体数据坐标系说明 (与后端 image_skills.py get_volume_raw_base64 完全一致):
 *   volume_data[x, y, z]  --降采样-->  resized_vol[96, 96, 96]
 *   z 轴 (dims[2]) = Axial    方向  (前端 axialIndex,    Three.js Y 轴, 头顶向上)
 *   y 轴 (dims[1]) = Coronal  方向  (前端 coronalIndex,  Three.js Z 轴)
 *   x 轴 (dims[0]) = Sagittal 方向  (前端 sagittalIndex, Three.js X 轴)
 *
 * Three.js 中切片按 Axial (z) 方向堆叠: PlaneGeometry 默认法线朝 +Z,
 * 旋转 -90° 绕 X 轴后法线朝 +Y (头顶向上), position.y 从 -50 到 +50。
 */
export default function Real3DViewer({
    axialIndex,
    coronalIndex,
    sagittalIndex,
    windowWidth = 7000,
    windowLevel = 3500,
    volumeThreshold = 0.20,
    showVolume = true,
    maskVersion = 'v0',
    labelLayers = []
}) {
    const mountRef = useRef(null);
    const rendererRef = useRef(null);
    const cameraRef = useRef(null);
    const sceneRef = useRef(null);
    const volumeGroupRef = useRef(null);
    const maskGroupRef = useRef(null);
    const planesGroupRef = useRef(null);
    const rawBytesRef = useRef(null);
    const dimsRef = useRef([96, 96, 96]);
    const rafIdRef = useRef(null);
    const [dataLoaded, setDataLoaded] = useState(false);
    const [statusMsg, setStatusMsg] = useState('⏳ 正在初始化 3D 渲染...');

    // ─── 1. 初始化 Scene ──────────────────────────────────────────────
    useEffect(() => {
        const mount = mountRef.current;
        if (!mount) return;

        // 必须在 mount 有实际像素后才能初始化，用 setTimeout 保证 DOM layout 完成
        const init = () => {
            const W = mount.clientWidth;
            const H = mount.clientHeight;
            if (W === 0 || H === 0) {
                // 容器尚未有尺寸，延迟重试
                setTimeout(init, 50);
                return;
            }

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0f172a);
            sceneRef.current = scene;

            const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 1000);
            camera.position.set(160, 120, 180);
            camera.lookAt(0, 0, 0);
            cameraRef.current = camera;

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(W, H);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            mount.appendChild(renderer.domElement);
            rendererRef.current = renderer;

            // 光照
            scene.add(new THREE.AmbientLight(0xffffff, 2.0));

            // 包围框
            const box = new THREE.BoxGeometry(100, 100, 100);
            scene.add(new THREE.LineSegments(
                new THREE.EdgesGeometry(box),
                new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.7 })
            ));

            // 地面网格 (Y = -50, 对应 Axial z=0 底部)
            const grid = new THREE.GridHelper(180, 12, 0x38bdf8, 0x334155);
            grid.position.y = -50;
            scene.add(grid);

            // 坐标轴 (原点在体数据 [0,0,0] 对应的 Three.js [-50,-50,-50])
            const axes = new THREE.AxesHelper(75);
            axes.position.set(-50, -50, -50);
            scene.add(axes);

            // 体数据组 & 准星组
            const volumeGroup = new THREE.Group();
            scene.add(volumeGroup);
            volumeGroupRef.current = volumeGroup;

            const maskGroup = new THREE.Group();
            scene.add(maskGroup);
            maskGroupRef.current = maskGroup;

            const planesGroup = new THREE.Group();
            scene.add(planesGroup);
            planesGroupRef.current = planesGroup;

            // 鼠标旋转
            let dragging = false;
            let prev = { x: 0, y: 0 };
            const onDown = (e) => { dragging = true; prev = { x: e.clientX, y: e.clientY }; };
            const onMove = (e) => {
                if (!dragging) return;
                scene.rotation.y += (e.clientX - prev.x) * 0.008;
                scene.rotation.x += (e.clientY - prev.y) * 0.008;
                prev = { x: e.clientX, y: e.clientY };
            };
            const onUp = () => { dragging = false; };
            renderer.domElement.addEventListener('mousedown', onDown);
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);

            // Resize
            const onResize = () => {
                const w = mount.clientWidth;
                const h = mount.clientHeight;
                if (w === 0 || h === 0) return;
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                renderer.setSize(w, h);
            };
            window.addEventListener('resize', onResize);

            // 渲染循环
            const animate = () => {
                rafIdRef.current = requestAnimationFrame(animate);
                renderer.render(scene, camera);
            };
            animate();

            // 拉取体数据
            fetch('/api/volume_data_raw')
                .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
                .then(data => {
                    if (!data?.raw_base64) { setStatusMsg('⚠️ 3D 数据为空'); return; }
                    const bin = window.atob(data.raw_base64);
                    const bytes = new Uint8Array(bin.length);
                    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                    rawBytesRef.current = bytes;
                    dimsRef.current = data.dimensions || [96, 96, 96];
                    setDataLoaded(true);
                    setStatusMsg('✨ 3D 体渲染已载入');
                })
                .catch(err => { setStatusMsg(`⚠️ 加载失败: ${err.message}`); });

            // 清理
            return () => {
                cancelAnimationFrame(rafIdRef.current);
                window.removeEventListener('resize', onResize);
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);
                renderer.domElement.removeEventListener('mousedown', onDown);
                renderer.dispose();
                if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
            };
        };

        const cleanup = init();
        return () => { if (typeof cleanup === 'function') cleanup(); };
    }, []);

    // ─── 2. 体数据切片栈 (WW/WL 驱动) ────────────────────────────────
    useEffect(() => {
        if (!dataLoaded || !rawBytesRef.current || !volumeGroupRef.current) return;

        const bytes = rawBytesRef.current;
        // dims = [dz, dx, dy]: dz=Axial切片数, dx=Sagittal像素, dy=Coronal像素
        const dims = dimsRef.current;
        const dz = dims[0];   // Axial 方向 → Three.js Y 轴 (头顶 = y=+50)
        const dx = dims[1];   // Sagittal 方向 → Three.js X 轴
        const dy = dims[2];   // Coronal 方向 → Three.js Z 轴
        const sliceSize = dx * dy;   // 每张 Axial 切片的字节数

        const group = volumeGroupRef.current;

        // 清空旧切片
        while (group.children.length > 0) {
            const c = group.children[0];
            c.material?.map?.dispose();
            c.material?.dispose();
            c.geometry?.dispose();
            group.remove(c);
        }
        if (!showVolume) return;

        // WW/WL 归一化 (后端原始数据已 P99 归一到 0-255)
        const normWW = Math.max(10, (windowWidth / 10000.0) * 255.0);
        const normWL = (windowLevel / 8000.0) * 255.0;
        const lo = Math.max(0, normWL - normWW / 2);
        const hi = Math.min(255, normWL + normWW / 2);
        const range = Math.max(1, hi - lo);
        const cutoff = volumeThreshold;

        // 每 STEP 张 Axial 切片构建一个实体 slab（BoxGeometry 有层厚）
        // slabH = 相邻切片在世界坐标 Y 轴上的间距 × 步长，紧密无缝拼接
        const STEP = 2;
        const slabH = (100 / (dz - 1)) * STEP;   // slab 在 Y 轴上的物理高度

        // 侧面共用的深色半透明材质（避免每个 slab 重复创建）
        const sideMat = new THREE.MeshBasicMaterial({
            color: 0x1e293b,
            transparent: true,
            opacity: 0.18,
            depthWrite: false,
            side: THREE.FrontSide
        });

        for (let zi = 0; zi < dz; zi += STEP) {
            const sliceData = bytes.subarray(zi * sliceSize, (zi + 1) * sliceSize);

            let hasBrain = false;
            for (let i = 0; i < sliceData.length; i += 4) {
                if (sliceData[i] > 20) { hasBrain = true; break; }
            }
            if (!hasBrain) continue;

            const rgba = new Uint8Array(sliceSize * 4);
            for (let i = 0; i < sliceSize; i++) {
                const raw = sliceData[i];
                const clipped = Math.max(lo, Math.min(hi, raw));
                const v = Math.floor(((clipped - lo) / range) * 255);
                const above = (v / 255) >= cutoff;
                rgba[i * 4 + 0] = v;
                rgba[i * 4 + 1] = v;
                rgba[i * 4 + 2] = v;
                rgba[i * 4 + 3] = (above && v > 15) ? Math.min(255, v * 2) : 0;
            }

            // DataTexture: width=dy(Coronal列), height=dx(Sagittal行)
            const tex = new THREE.DataTexture(rgba, dy, dx, THREE.RGBAFormat, THREE.UnsignedByteType);
            tex.needsUpdate = true;

            // 顶底面贴 MRI 纹理的材质
            const faceMat = new THREE.MeshBasicMaterial({
                map: tex,
                transparent: true,
                opacity: 0.30,
                depthWrite: false,
                side: THREE.DoubleSide
            });

            // BoxGeometry: 宽100(X/Sagittal) × 高slabH(Y/Axial层厚) × 深100(Z/Coronal)
            // material 顺序: [+X,-X,+Y,-Y,+Z,-Z] → 顶底(2,3)贴纹理，四侧(0,1,4,5)深色
            const geo = new THREE.BoxGeometry(100, slabH, 100);
            const mesh = new THREE.Mesh(geo, [
                sideMat, sideMat,   // +X / -X 侧面
                faceMat, faceMat,   // +Y / -Y 顶底面 ← MRI 纹理
                sideMat, sideMat    // +Z / -Z 侧面
            ]);

            // BoxGeometry 已是轴对齐，无需旋转；Y 位置 = slab 中心
            mesh.position.y = (zi / (dz - 1)) * 100 - 50;


            group.add(mesh);
        }
    }, [dataLoaded, windowWidth, windowLevel, volumeThreshold, showVolume]);

    // ─── 3. 正交准星指示线 ────────────────────────────────────────────
    useEffect(() => {
        const group = planesGroupRef.current;
        if (!group) return;
        while (group.children.length > 0) group.remove(group.children[0]);

        // 坐标系映射 (与切片一致):
        //   axialIndex    (0~181) → Three.js Y 轴 (-50 ~ +50)
        //   coronalIndex  (0~217) → Three.js Z 轴 (-50 ~ +50)
        //   sagittalIndex (0~181) → Three.js X 轴 (-50 ~ +50)
        const posY = (axialIndex / 181) * 100 - 50;
        const posZ = (coronalIndex / 217) * 100 - 50;
        const posX = (sagittalIndex / 181) * 100 - 50;

        const addLine = (p1, p2, color) => {
            const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...p1), new THREE.Vector3(...p2)]);
            group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color })));
        };

        // Axial 准星: 显示当前 axial 切片所在的水平面位置 (水平线)
        addLine([-50, posY, 0], [50, posY, 0], 0x38bdf8);   // X 方向蓝线
        addLine([0, posY, -50], [0, posY, 50], 0x38bdf8);   // Z 方向蓝线

        // Coronal 准星: 显示当前 coronal 切片 (垂直面 Z)
        addLine([-50, -50, posZ], [50, 50, posZ], 0x10b981);
        addLine([posX, -50, posZ], [posX, 50, posZ], 0x10b981);

        // Sagittal 准星: 显示当前 sagittal 切片 (垂直面 X)
        addLine([posX, -50, -50], [posX, 50, 50], 0xef4444);
        addLine([posX, -50, posZ], [posX, 50, posZ], 0xef4444);

    }, [axialIndex, coronalIndex, sagittalIndex]);

    // ─── 4. 3D Mask 重建 ─────────────────────────────────────────────
    useEffect(() => {
        const group = maskGroupRef.current;
        if (!group) return;

        // 清空旧 mask mesh
        while (group.children.length > 0) {
            const c = group.children[0];
            c.material?.map?.dispose();
            c.material?.dispose();
            c.geometry?.dispose();
            group.remove(c);
        }

        // 找到当前可见图层的颜色
        const activeLayer = labelLayers.find(l => l.visible);
        if (!activeLayer) return;

        // 解析颜色
        const hexColor = activeLayer.color || '#06b6d4';
        const r = parseInt(hexColor.slice(1, 3), 16) / 255;
        const g = parseInt(hexColor.slice(3, 5), 16) / 255;
        const b = parseInt(hexColor.slice(5, 7), 16) / 255;
        const color = new THREE.Color(r, g, b);

        fetch('/api/mask_volume_raw')
            .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
            .then(data => {
                if (!data?.raw_base64 || !data.has_mask) return;

                const bin = window.atob(data.raw_base64);
                const bytes = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

                const dims = data.dimensions;  // [dz, dy, dx]
                const dz = dims[0], dy = dims[1], dx = dims[2];
                const sliceSize = dy * dx;

                const STEP = 2;
                const slabH = (100 / (dz - 1)) * STEP;

                for (let zi = 0; zi < dz; zi += STEP) {
                    const sliceData = bytes.subarray(zi * sliceSize, (zi + 1) * sliceSize);

                    let hasMask = false;
                    for (let i = 0; i < sliceData.length; i++) {
                        if (sliceData[i] > 0) { hasMask = true; break; }
                    }
                    if (!hasMask) continue;

                    const rgba = new Uint8Array(sliceSize * 4);
                    for (let i = 0; i < sliceSize; i++) {
                        const v = sliceData[i] > 0 ? 255 : 0;
                        rgba[i * 4 + 0] = v;
                        rgba[i * 4 + 1] = v;
                        rgba[i * 4 + 2] = v;
                        rgba[i * 4 + 3] = v > 0 ? 200 : 0;
                    }

                    const tex = new THREE.DataTexture(rgba, dx, dy, THREE.RGBAFormat, THREE.UnsignedByteType);
                    tex.needsUpdate = true;

                    const faceMat = new THREE.MeshBasicMaterial({
                        map: tex,
                        color,
                        transparent: true,
                        opacity: activeLayer.opacity ?? 0.6,
                        depthWrite: false,
                        side: THREE.DoubleSide
                    });

                    // 侧面用纯色（与顶底面一致颜色，低透明度）
                    const sideMat = new THREE.MeshBasicMaterial({
                        color,
                        transparent: true,
                        opacity: (activeLayer.opacity ?? 0.6) * 0.4,
                        depthWrite: false,
                        side: THREE.FrontSide
                    });

                    const geo = new THREE.BoxGeometry(100, slabH, 100);
                    const mesh = new THREE.Mesh(geo, [
                        sideMat, sideMat,
                        faceMat, faceMat,
                        sideMat, sideMat
                    ]);
                    // 略微抬高 0.15 避免与体数据 z-fighting
                    mesh.position.y = (zi / (dz - 1)) * 100 - 50 + 0.15;
                    group.add(mesh);
                }

            })
            .catch(err => console.warn('3D Mask 加载失败:', err));

    }, [maskVersion, labelLayers]);

    return (
        <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: '#0f172a' }}>
            <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
            <div style={{
                position: 'absolute', bottom: 8, left: 10,
                fontSize: 10, color: '#94a3b8',
                background: 'rgba(0,0,0,0.7)', padding: '3px 8px',
                borderRadius: 4, pointerEvents: 'none',
                display: 'flex', gap: 8, alignItems: 'center'
            }}>
                <span>{statusMsg}</span>
                <span style={{ color: '#ef4444' }}>■ X 矢状</span>
                <span style={{ color: '#10b981' }}>■ Z 冠状</span>
                <span style={{ color: '#38bdf8' }}>■ Y 轴位/头顶</span>
            </div>
        </div>
    );
}
