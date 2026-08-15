import React, { useEffect, useRef, useState } from 'react';
import '@kitware/vtk.js/Rendering/Profiles/All';
import '@kitware/vtk.js/Rendering/Profiles/Volume';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
import vtkOrientationMarkerWidget from '@kitware/vtk.js/Interaction/Widgets/OrientationMarkerWidget';
import vtkAxesActor from '@kitware/vtk.js/Rendering/Core/AxesActor';
import vtkInteractorStyleTrackballCamera from '@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera';

/**
 * Vtk3DVolumeViewer - 基于 @kitware/vtk.js 的原生 3D GPU Ray-Marching 体渲染视口
 */
export default function Vtk3DVolumeViewer({
    volumeData = null,
    maskData = null,
    windowWidth = 7000,
    windowLevel = 3500,
    volumeThreshold = 0.20,
    showVolume = true,
    activeLayer = null
}) {
    const containerRef = useRef(null);
    const grwRef = useRef(null);
    const volumeRef = useRef(null);
    const volumeMapperRef = useRef(null);
    const maskVolumeRef = useRef(null);
    const maskVolumeMapperRef = useRef(null);
    const cfunRef = useRef(null);
    const ofunRef = useRef(null);
    const maskCfunRef = useRef(null);
    const maskOfunRef = useRef(null);
    const orientationWidgetRef = useRef(null);
    const [statusMsg, setStatusMsg] = useState('GPU Ray Marching + SSAO/LAO (Active)');

    // 1. 初始化 3D WebGL GenericRenderWindow 与独立子 Renderer 坐标轴系统 (仅在 volumeData 有效时)
    useEffect(() => {
        const container = containerRef.current;
        if (!container || !volumeData) return;

        const grw = vtkGenericRenderWindow.newInstance({
            background: [1.0, 1.0, 1.0] // 纯白高精背景 (#ffffff)
        });
        grw.setContainer(container);
        grwRef.current = grw;

        const renderer = grw.getRenderer();
        renderer.setBackground(1.0, 1.0, 1.0);
        const renderWindow = grw.getRenderWindow();
        const interactor = grw.getInteractor();

        // 1.1 主 3D MRI Volume 管道 (光追体积环境遮蔽 Volumetric SSAO + SSS 组织散射)
        const volume = vtkVolume.newInstance();
        const volumeMapper = vtkVolumeMapper.newInstance();
        volumeMapper.setSampleDistance(0.4); // 高密度光线投射步长，消除阶梯噪点，模拟平滑体积散射
        volumeMapper.setAutoAdjustSampleDistances(false);
        volumeMapper.setMaximumSamplesPerRay(2000);
        volumeMapper.setInputData(volumeData);
        volume.setMapper(volumeMapper);
        volume.setVisibility(showVolume);

        const vprop = volume.getProperty();
        const cfun = vtkColorTransferFunction.newInstance();
        const ofun = vtkPiecewiseFunction.newInstance();
        vprop.setRGBTransferFunction(0, cfun);
        vprop.setScalarOpacity(0, ofun);
        vprop.setInterpolationTypeToLinear();
        vprop.setShade(true);
        // SSS 模拟光照：高环境光(内透光肉感) + 柔漫射 + 宽泛生物湿润微光
        vprop.setAmbient(0.44);
        vprop.setDiffuse(0.68);
        vprop.setSpecular(0.18);
        vprop.setSpecularPower(10.0);

        // 开启光线追踪体积环境光遮蔽 (Volumetric SSAO / Local Ambient Occlusion)
        vprop.setLocalAmbientOcclusion(true);
        vprop.setLAOKernelSize(11);
        vprop.setLAOKernelRadius(5.0);
        vprop.setVolumetricScatteringBlending(0.35);

        // 启用梯度不透明度 (Gradient Opacity) 实现组织内透与沟回轮廓散射
        vprop.setUseGradientOpacity(0, true);
        vprop.setGradientOpacityMinimumValue(0, 1.2);
        vprop.setGradientOpacityMinimumOpacity(0, 0.15);
        vprop.setGradientOpacityMaximumValue(0, 14.0);
        vprop.setGradientOpacityMaximumOpacity(0, 1.0);

        volumeRef.current = volume;
        volumeMapperRef.current = volumeMapper;
        cfunRef.current = cfun;
        ofunRef.current = ofun;
        renderer.addVolume(volume);

        // 1.2 3D Mask 分割体渲染管道 (开启环境遮蔽与平滑光照)
        const maskVolume = vtkVolume.newInstance();
        const maskVolumeMapper = vtkVolumeMapper.newInstance();
        maskVolumeMapper.setSampleDistance(0.5);
        maskVolumeMapper.setAutoAdjustSampleDistances(false);
        maskVolumeMapper.setMaximumSamplesPerRay(1500);
        if (maskData) {
            maskVolumeMapper.setInputData(maskData);
            maskVolume.setVisibility(Boolean(activeLayer && activeLayer.visible));
        } else {
            maskVolume.setVisibility(false);
        }
        maskVolume.setMapper(maskVolumeMapper);

        const maskCfun = vtkColorTransferFunction.newInstance();
        const maskOfun = vtkPiecewiseFunction.newInstance();
        maskVolume.getProperty().setRGBTransferFunction(0, maskCfun);
        maskVolume.getProperty().setScalarOpacity(0, maskOfun);
        maskVolume.getProperty().setInterpolationTypeToLinear();
        maskVolume.getProperty().setShade(true);
        maskVolume.getProperty().setAmbient(0.45);
        maskVolume.getProperty().setDiffuse(0.75);
        maskVolume.getProperty().setSpecular(0.20);
        maskVolume.getProperty().setSpecularPower(12.0);
        maskVolume.getProperty().setLocalAmbientOcclusion(true);
        maskVolume.getProperty().setLAOKernelSize(9);
        maskVolume.getProperty().setLAOKernelRadius(4.0);

        maskVolumeRef.current = maskVolume;
        maskVolumeMapperRef.current = maskVolumeMapper;
        maskCfunRef.current = maskCfun;
        maskOfunRef.current = maskOfun;
        renderer.addVolume(maskVolume);

        // 1.3 核心防覆写架构：独立子 Renderer (扁平纯色无光照 AxesActor + OrientationMarkerWidget)
        let orientationWidget = null;
        try {
            const axes = vtkAxesActor.newInstance({
                config: {
                    recenter: true,
                    tipRadius: 0.11,
                    tipLength: 0.22,
                    shaftRadius: 0.035
                },
                xConfig: { color: [225, 29, 72] }, // Rose 600 (#e11d48) - 矢状位
                yConfig: { color: [217, 119, 6] }, // Amber 600 (#d97706) - 冠状位
                zConfig: { color: [22, 163, 74] }  // Green 600 (#16a34a) - 轴位
            });
            axes.setXAxisColor([225, 29, 72]);
            axes.setYAxisColor([217, 119, 6]);
            axes.setZAxisColor([22, 163, 74]);

            // 关闭 3D PBR 与 Phong 光照，对齐扁平 Flat Design 设计风格
            const prop = axes.getProperty();
            if (prop) {
                prop.setLighting(false);
                prop.setAmbient(1.0);
                prop.setDiffuse(0.0);
                prop.setSpecular(0.0);
            }

            orientationWidget = vtkOrientationMarkerWidget.newInstance();
            orientationWidget.setActor(axes);
            orientationWidget.setParentRenderer(renderer);
            orientationWidget.setInteractor(interactor);
            if (vtkOrientationMarkerWidget.Corners) {
                orientationWidget.setViewportCorner(vtkOrientationMarkerWidget.Corners.BOTTOM_RIGHT);
            }
            orientationWidget.setViewportSize(0.2);
            orientationWidgetRef.current = orientationWidget;
        } catch (e) {
            console.warn('初始化 OrientationMarkerWidget 异常:', e);
        }

        // 1.4 初始化 3D 交互器样式 (支持左键旋转、右键缩放、中键平移 Pan、Shift+左键平移)
        if (interactor) {
            const iStyle = vtkInteractorStyleTrackballCamera.newInstance();
            interactor.setInteractorStyle(iStyle);
            interactor.initialize();
            interactor.bindEvents(container);
            if (orientationWidget) {
                try {
                    orientationWidget.setEnabled(true);
                } catch (e) {
                    console.warn('启用 OrientationMarkerWidget 失败:', e);
                }
            }
        }

        // 1.5 设置标准 3D 轴测视角 (Z+ 头顶向上, Y+ 朝左, X+ 朝左)
        const bounds = volumeData.getBounds();
        const center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ];
        const camera = renderer.getActiveCamera();
        camera.setFocalPoint(center[0], center[1], center[2]);

        const dist = 380;
        camera.setPosition(
            center[0] - dist * 0.60, // 位于 -X 侧 (视线指向 +X，保证 Y+ 轴投影朝左)
            center[1] + dist * 0.70, // 位于 +Y 侧 (视线指向 -Y，保证 X+ 轴投影朝左)
            center[2] + dist * 0.50  // 位于 +Z 侧 (保证俯视立体轴测感)
        );
        camera.setViewUp(0, 0, 1);   // Z+ (Superior / 头顶) 严格朝上
        renderer.resetCamera();
        camera.zoom(1.2);

        // 原生平滑滚轮缩放
        const handleWheel = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
            camera.zoom(zoomFactor);
            renderWindow.render();
        };

        // 阻止浏览器对鼠标中键点击的默认滚轮行为，确保中键拖拽平移 100% 顺畅
        const handleAuxClick = (e) => {
            if (e.button === 1) e.preventDefault();
        };

        container.addEventListener('wheel', handleWheel, { passive: false });
        container.addEventListener('auxclick', handleAuxClick);

        const handleResize = () => {
            if (!container) return;
            grw.resize();
            renderWindow.render();
        };
        window.addEventListener('resize', handleResize);
        handleResize();

        return () => {
            window.removeEventListener('resize', handleResize);
            container.removeEventListener('wheel', handleWheel);
            container.removeEventListener('auxclick', handleAuxClick);
            if (orientationWidget) {
                try {
                    orientationWidget.setEnabled(false);
                    orientationWidget.delete();
                } catch (e) {}
            }
            if (interactor) {
                interactor.unbindEvents();
            }
            grw.delete();
        };
    }, [volumeData]);

    // 2. 响应主体数据显隐与 WW/WL / ISO 阈值传递函数
    useEffect(() => {
        if (!cfunRef.current || !ofunRef.current || !volumeRef.current || !grwRef.current || !volumeData) return;

        const cfun = cfunRef.current;
        const ofun = ofunRef.current;
        const volume = volumeRef.current;
        const renderWindow = grwRef.current.getRenderWindow();

        volume.setVisibility(showVolume);
        if (!showVolume) {
            renderWindow.render();
            return;
        }

        // WW/WL 映射到 0-255 标量
        const normWW = Math.max(10.0, (windowWidth / 10000.0) * 255.0);
        const normWL = (windowLevel / 8000.0) * 255.0;

        const minVal = Math.max(0, normWL - normWW / 2.0);
        const maxVal = Math.min(255, normWL + normWW / 2.0);
        const cutoff = Math.max(5, minVal + (maxVal - minVal) * volumeThreshold);

        // SSS 级生物组织体积散射颜色传递函数 (Subsurface Scattering Spectrum)
        cfun.removeAllPoints();
        cfun.addRGBPoint(0, 1.0, 1.0, 1.0);
        cfun.addRGBPoint(cutoff, 0.90, 0.62, 0.58);                         // 浅表微血管/脑膜边缘: 透光珊瑚粉
        cfun.addRGBPoint(cutoff + (maxVal - cutoff) * 0.20, 0.85, 0.52, 0.48); // 大脑皮层浅层灰质: 鲜活肉粉色
        cfun.addRGBPoint(cutoff + (maxVal - cutoff) * 0.45, 0.80, 0.56, 0.52); // 大脑皮层深层灰质: 温润脑回实质色
        cfun.addRGBPoint(cutoff + (maxVal - cutoff) * 0.70, 0.96, 0.90, 0.82); // 大脑白质/深部髓质: 象牙暖米白
        cfun.addRGBPoint(maxVal, 0.98, 0.94, 0.88);                         // 致密硬脑膜与骨质: 象牙硬骨色
        cfun.addRGBPoint(255, 1.0, 0.96, 0.90);

        // SSS 柔和吸收率曲线：浅表半透内透光，深部致密
        ofun.removeAllPoints();
        ofun.addPoint(0, 0.0);
        ofun.addPoint(cutoff, 0.0);
        ofun.addPoint(cutoff + (maxVal - cutoff) * 0.12, 0.08); // 浅表半透产生内散射光晕
        ofun.addPoint(cutoff + (maxVal - cutoff) * 0.30, 0.32);
        ofun.addPoint(cutoff + (maxVal - cutoff) * 0.55, 0.65);
        ofun.addPoint(cutoff + (maxVal - cutoff) * 0.80, 0.88);
        ofun.addPoint(maxVal, 0.94);
        ofun.addPoint(255, 0.98);

        renderWindow.render();
    }, [windowWidth, windowLevel, volumeThreshold, showVolume, volumeData]);

    // 3. 响应 3D Mask 分割颜色与透明度
    useEffect(() => {
        if (!maskCfunRef.current || !maskOfunRef.current || !maskVolumeRef.current || !maskVolumeMapperRef.current || !grwRef.current || !volumeData) return;

        const maskMapper = maskVolumeMapperRef.current;
        const maskVolume = maskVolumeRef.current;
        const maskCfun = maskCfunRef.current;
        const maskOfun = maskOfunRef.current;
        const renderWindow = grwRef.current.getRenderWindow();

        if (maskData && activeLayer && activeLayer.visible) {
            maskMapper.setInputData(maskData);
            maskVolume.setVisibility(true);

            const hex = activeLayer.color || '#06b6d4';
            const r = parseInt(hex.slice(1, 3), 16) / 255;
            const g = parseInt(hex.slice(3, 5), 16) / 255;
            const b = parseInt(hex.slice(5, 7), 16) / 255;

            maskCfun.removeAllPoints();
            maskCfun.addRGBPoint(0, 0.0, 0.0, 0.0);
            maskCfun.addRGBPoint(1, r, g, b);
            maskCfun.addRGBPoint(255, r, g, b);

            const opacity = (activeLayer.opacity ?? 0.6) * 1.2;
            maskOfun.removeAllPoints();
            maskOfun.addPoint(0, 0.0);
            maskOfun.addPoint(1, Math.min(1.0, opacity));
            maskOfun.addPoint(255, Math.min(1.0, opacity));
        } else {
            maskVolume.setVisibility(false);
        }

        renderWindow.render();
    }, [activeLayer, maskData, volumeData]);

    if (!volumeData) {
        return (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#ffffff', color: '#94a3b8', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                <span>INITIALIZING VTK 3D RAY MARCHING...</span>
            </div>
        );
    }

    return (
        <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: '#ffffff' }}>
            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
            <div style={{
                position: 'absolute', bottom: 10, left: 12,
                fontSize: 10.5, color: '#334155',
                background: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)',
                padding: '4px 10px',
                borderRadius: 4, pointerEvents: 'none',
                display: 'flex', gap: 12, alignItems: 'center',
                zIndex: 10, border: '1px solid #e2e8f0',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)'
            }}>
                <span style={{ color: '#64748b', fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>{statusMsg}</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#e11d48', fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>
                    <span style={{ width: 6, height: 6, borderRadius: 1, backgroundColor: '#e11d48', display: 'inline-block' }}></span>
                    <span>X: 矢状</span>
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#d97706', fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>
                    <span style={{ width: 6, height: 6, borderRadius: 1, backgroundColor: '#d97706', display: 'inline-block' }}></span>
                    <span>Y: 冠状</span>
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#16a34a', fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>
                    <span style={{ width: 6, height: 6, borderRadius: 1, backgroundColor: '#16a34a', display: 'inline-block' }}></span>
                    <span>Z: 轴位</span>
                </span>
            </div>
        </div>
    );
}
