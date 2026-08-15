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
    const [statusMsg, setStatusMsg] = useState('GPU Ray Marching Engine (Active)');

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

        // 1.1 主 3D MRI Volume 管道
        const volume = vtkVolume.newInstance();
        const volumeMapper = vtkVolumeMapper.newInstance();
        volumeMapper.setSampleDistance(1.0);
        volumeMapper.setInputData(volumeData);
        volume.setMapper(volumeMapper);
        volume.setVisibility(showVolume);

        const cfun = vtkColorTransferFunction.newInstance();
        const ofun = vtkPiecewiseFunction.newInstance();
        volume.getProperty().setRGBTransferFunction(0, cfun);
        volume.getProperty().setScalarOpacity(0, ofun);
        volume.getProperty().setInterpolationTypeToLinear();
        volume.getProperty().setShade(true);
        volume.getProperty().setAmbient(0.35);
        volume.getProperty().setDiffuse(0.65);
        volume.getProperty().setSpecular(0.40);
        volume.getProperty().setSpecularPower(25.0);

        volumeRef.current = volume;
        volumeMapperRef.current = volumeMapper;
        cfunRef.current = cfun;
        ofunRef.current = ofun;
        renderer.addVolume(volume);

        // 1.2 3D Mask 分割体渲染管道
        const maskVolume = vtkVolume.newInstance();
        const maskVolumeMapper = vtkVolumeMapper.newInstance();
        maskVolumeMapper.setSampleDistance(1.0);
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
        maskVolume.getProperty().setInterpolationTypeToNearest();
        maskVolume.getProperty().setShade(true);
        maskVolume.getProperty().setAmbient(0.4);
        maskVolume.getProperty().setDiffuse(0.8);

        maskVolumeRef.current = maskVolume;
        maskVolumeMapperRef.current = maskVolumeMapper;
        maskCfunRef.current = maskCfun;
        maskOfunRef.current = maskOfun;
        renderer.addVolume(maskVolume);

        // 1.3 核心防覆写架构：独立子 Renderer (OrientationMarkerWidget + AxesActor)
        let orientationWidget = null;
        try {
            const axes = vtkAxesActor.newInstance();
            orientationWidget = vtkOrientationMarkerWidget.newInstance();
            orientationWidget.setActor(axes);
            orientationWidget.setParentRenderer(renderer);
            orientationWidget.setInteractor(interactor);
            if (vtkOrientationMarkerWidget.Corners) {
                orientationWidget.setViewportCorner(vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT);
            }
            orientationWidget.setViewportSize(0.2);
            orientationWidgetRef.current = orientationWidget;
        } catch (e) {
            console.warn('初始化 OrientationMarkerWidget 异常:', e);
        }

        // 1.4 初始化 Interactor 与使能 Widget
        if (interactor) {
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

        const handleResize = () => {
            if (!container) return;
            grw.resize();
            renderWindow.render();
        };
        window.addEventListener('resize', handleResize);
        handleResize();

        return () => {
            window.removeEventListener('resize', handleResize);
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

        // 真实人体脑组织颜色传递函数 (Realistic Anatomical Color Transfer Function)
        cfun.removeAllPoints();
        cfun.addRGBPoint(0, 1.0, 1.0, 1.0);
        cfun.addRGBPoint(cutoff, 0.88, 0.65, 0.60); // 脑脊液/皮层边缘: 浅珊瑚肉粉
        cfun.addRGBPoint(cutoff + (maxVal - cutoff) * 0.25, 0.82, 0.54, 0.50); // 大脑皮层灰质: 真实灰质肉粉色
        cfun.addRGBPoint(cutoff + (maxVal - cutoff) * 0.60, 0.96, 0.90, 0.82); // 大脑白质: 象牙暖白髓质色
        cfun.addRGBPoint(maxVal, 0.98, 0.94, 0.88); // 颅底与硬膜致密组织: 象牙硬骨色
        cfun.addRGBPoint(255, 1.0, 0.96, 0.90);

        ofun.removeAllPoints();
        ofun.addPoint(0, 0.0);
        ofun.addPoint(cutoff, 0.0);
        ofun.addPoint(cutoff + (maxVal - cutoff) * 0.15, 0.15);
        ofun.addPoint(cutoff + (maxVal - cutoff) * 0.35, 0.45);
        ofun.addPoint(cutoff + (maxVal - cutoff) * 0.70, 0.80);
        ofun.addPoint(maxVal, 0.92);
        ofun.addPoint(255, 0.95);

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
