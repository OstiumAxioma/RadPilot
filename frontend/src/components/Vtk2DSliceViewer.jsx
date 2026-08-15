import React, { useEffect, useRef } from 'react';
import '@kitware/vtk.js/Rendering/Profiles/All';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkImageSlice from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkImageMapper from '@kitware/vtk.js/Rendering/Core/ImageMapper';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';

/**
 * Vtk2DSliceViewer - 基于 @kitware/vtk.js 的原生正交切片视口
 * @param {string} axis - 'axial' | 'coronal' | 'sagittal'
 * @param {number} sliceIndex - 当前切片索引
 * @param {vtkImageData} volumeData - 主 MRI vtkImageData (非空时才挂载 VTK 管线)
 * @param {vtkImageData} maskData - 分割 Mask vtkImageData
 * @param {number} windowWidth - 窗宽 (WW)
 * @param {number} windowLevel - 窗位 (WL)
 * @param {object} activeLayer - 当前可见的图层配置 { color, opacity, visible }
 */
export default function Vtk2DSliceViewer({
    axis = 'axial',
    sliceIndex = 90,
    volumeData = null,
    maskData = null,
    windowWidth = 7000,
    windowLevel = 3500,
    activeLayer = null
}) {
    const containerRef = useRef(null);
    const grwRef = useRef(null);
    const imageSliceRef = useRef(null);
    const imageMapperRef = useRef(null);
    const maskSliceRef = useRef(null);
    const maskMapperRef = useRef(null);
    const cfunRef = useRef(null);
    const ofunRef = useRef(null);
    const maskCfunRef = useRef(null);
    const maskOfunRef = useRef(null);

    // 1. 初始化 VTK GenericRenderWindow 与切片管道 (仅在 volumeData 存在时初始化)
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
        const camera = renderer.getActiveCamera();
        camera.setParallelProjection(true); // 2D 医学切片严格使用平行正交投影

        // 1.1 主 MRI 切片管道
        const imageSlice = vtkImageSlice.newInstance();
        const imageMapper = vtkImageMapper.newInstance();
        imageMapper.setInputData(volumeData);
        imageSlice.setMapper(imageMapper);

        const cfun = vtkColorTransferFunction.newInstance();
        const ofun = vtkPiecewiseFunction.newInstance();
        imageSlice.getProperty().setRGBTransferFunction(cfun);
        imageSlice.getProperty().setPiecewiseFunction(ofun);

        imageSliceRef.current = imageSlice;
        imageMapperRef.current = imageMapper;
        cfunRef.current = cfun;
        ofunRef.current = ofun;
        renderer.addActor(imageSlice);

        // 1.2 Mask 叠加切片管道
        const maskSlice = vtkImageSlice.newInstance();
        const maskMapper = vtkImageMapper.newInstance();
        if (maskData) {
            maskMapper.setInputData(maskData);
            maskSlice.setVisibility(Boolean(activeLayer && activeLayer.visible));
        } else {
            maskSlice.setVisibility(false);
        }
        maskSlice.setMapper(maskMapper);

        const maskCfun = vtkColorTransferFunction.newInstance();
        const maskOfun = vtkPiecewiseFunction.newInstance();
        maskSlice.getProperty().setRGBTransferFunction(maskCfun);
        maskSlice.getProperty().setPiecewiseFunction(maskOfun);

        maskSliceRef.current = maskSlice;
        maskMapperRef.current = maskMapper;
        maskCfunRef.current = maskCfun;
        maskOfunRef.current = maskOfun;
        renderer.addActor(maskSlice);

        // 1.3 设置切片模式与初始切片
        if (axis === 'axial') {
            imageMapper.setSlicingMode(vtkImageMapper.SlicingMode.K);
            maskMapper.setSlicingMode(vtkImageMapper.SlicingMode.K);
            imageMapper.setKSlice(sliceIndex);
            if (maskData) maskMapper.setKSlice(sliceIndex);
        } else if (axis === 'coronal') {
            imageMapper.setSlicingMode(vtkImageMapper.SlicingMode.J);
            maskMapper.setSlicingMode(vtkImageMapper.SlicingMode.J);
            imageMapper.setJSlice(sliceIndex);
            if (maskData) maskMapper.setJSlice(sliceIndex);
        } else {
            imageMapper.setSlicingMode(vtkImageMapper.SlicingMode.I);
            maskMapper.setSlicingMode(vtkImageMapper.SlicingMode.I);
            imageMapper.setISlice(sliceIndex);
            if (maskData) maskMapper.setISlice(sliceIndex);
        }

        // 1.4 设置相机观察方向与中心
        const bounds = volumeData.getBounds();
        const center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ];
        camera.setFocalPoint(center[0], center[1], center[2]);

        const dist = 500;
        if (axis === 'axial') {
            camera.setPosition(center[0], center[1], center[2] + dist);
            camera.setViewUp(0, 1, 0); // Anterior 向上
        } else if (axis === 'coronal') {
            camera.setPosition(center[0], center[1] - dist, center[2]);
            camera.setViewUp(0, 0, 1); // Superior(头顶) 向上
        } else {
            // 从左向右看 (X轴侧视) -> 屏幕左侧为 +Y (Anterior/鼻子)，右侧为 -Y (Posterior/脑勺)，上方为 +Z (Superior/头顶)
            camera.setPosition(center[0] - dist, center[1], center[2]);
            camera.setViewUp(0, 0, 1); // Superior(头顶) 向上
        }
        renderer.resetCamera();

        // 1.5 初始反相上色 (Inverted Positive Transfer Function: 0=白, 高信号=深黑灰)
        const normWW = Math.max(5.0, (windowWidth / 10000.0) * 255.0);
        const normWL = (windowLevel / 8000.0) * 255.0;
        const lower = Math.max(0, normWL - normWW / 2.0);
        const upper = Math.min(255, normWL + normWW / 2.0);

        cfun.removeAllPoints();
        cfun.addRGBPoint(0, 1.0, 1.0, 1.0);
        cfun.addRGBPoint(lower, 1.0, 1.0, 1.0);
        cfun.addRGBPoint(upper, 0.08, 0.08, 0.10);
        cfun.addRGBPoint(255, 0.05, 0.05, 0.06);

        ofun.removeAllPoints();
        ofun.addPoint(0, 1.0);
        ofun.addPoint(255, 1.0);

        // 1.6 初始化交互器与 Resize
        const interactor = grw.getInteractor();
        if (interactor) {
            interactor.initialize();
            interactor.bindEvents(container);
        }

        const handleResize = () => {
            if (!container) return;
            grw.resize();
            renderWindow.render();
        };
        window.addEventListener('resize', handleResize);
        handleResize();

        return () => {
            window.removeEventListener('resize', handleResize);
            if (interactor) {
                interactor.unbindEvents();
            }
            grw.delete();
        };
    }, [axis, volumeData]);

    // 2. 更新切片位置 (Slice Index)
    useEffect(() => {
        if (!imageMapperRef.current || !grwRef.current || !volumeData) return;

        const imageMapper = imageMapperRef.current;
        const maskMapper = maskMapperRef.current;
        const renderWindow = grwRef.current.getRenderWindow();

        if (axis === 'axial') {
            imageMapper.setKSlice(sliceIndex);
            if (maskMapper && maskData) maskMapper.setKSlice(sliceIndex);
        } else if (axis === 'coronal') {
            imageMapper.setJSlice(sliceIndex);
            if (maskMapper && maskData) maskMapper.setJSlice(sliceIndex);
        } else {
            imageMapper.setISlice(sliceIndex);
            if (maskMapper && maskData) maskMapper.setISlice(sliceIndex);
        }

        renderWindow.render();
    }, [sliceIndex, axis, volumeData, maskData]);

    // 3. 更新 WW/WL 窗宽窗位
    useEffect(() => {
        if (!cfunRef.current || !ofunRef.current || !grwRef.current || !volumeData) return;

        const cfun = cfunRef.current;
        const ofun = ofunRef.current;
        const renderWindow = grwRef.current.getRenderWindow();

        const normWW = Math.max(5.0, (windowWidth / 10000.0) * 255.0);
        const normWL = (windowLevel / 8000.0) * 255.0;

        const lower = Math.max(0, normWL - normWW / 2.0);
        const upper = Math.min(255, normWL + normWW / 2.0);

        cfun.removeAllPoints();
        cfun.addRGBPoint(0, 1.0, 1.0, 1.0);
        cfun.addRGBPoint(lower, 1.0, 1.0, 1.0);
        cfun.addRGBPoint(upper, 0.08, 0.08, 0.10);
        cfun.addRGBPoint(255, 0.05, 0.05, 0.06);

        ofun.removeAllPoints();
        ofun.addPoint(0, 1.0);
        ofun.addPoint(255, 1.0);

        renderWindow.render();
    }, [windowWidth, windowLevel, volumeData]);

    // 4. 更新 Mask 数据与色彩
    useEffect(() => {
        if (!maskMapperRef.current || !maskSliceRef.current || !grwRef.current || !volumeData) return;

        const maskMapper = maskMapperRef.current;
        const maskSlice = maskSliceRef.current;
        const maskCfun = maskCfunRef.current;
        const maskOfun = maskOfunRef.current;
        const renderWindow = grwRef.current.getRenderWindow();

        if (maskData && activeLayer && activeLayer.visible) {
            maskMapper.setInputData(maskData);
            maskSlice.setVisibility(true);

            const hex = activeLayer.color || '#06b6d4';
            const r = parseInt(hex.slice(1, 3), 16) / 255;
            const g = parseInt(hex.slice(3, 5), 16) / 255;
            const b = parseInt(hex.slice(5, 7), 16) / 255;

            if (maskCfun) {
                maskCfun.removeAllPoints();
                maskCfun.addRGBPoint(0, 0, 0, 0);
                maskCfun.addRGBPoint(1, r, g, b);
                maskCfun.addRGBPoint(255, r, g, b);
            }

            if (maskOfun) {
                const opacity = activeLayer.opacity ?? 0.6;
                maskOfun.removeAllPoints();
                maskOfun.addPoint(0, 0.0);
                maskOfun.addPoint(1, opacity);
                maskOfun.addPoint(255, opacity);
            }
        } else {
            maskSlice.setVisibility(false);
        }

        renderWindow.render();
    }, [maskData, activeLayer, volumeData]);

    // 解剖方位指示字符
    const getLabels = () => {
        if (axis === 'axial') return { top: 'A (前)', bottom: 'P (后)', left: 'R (右)', right: 'L (左)' };
        if (axis === 'coronal') return { top: 'S (头顶)', bottom: 'I (脚底)', left: 'R (右)', right: 'L (左)' };
        return { top: 'S (头顶)', bottom: 'I (脚底)', left: 'A (前/鼻)', right: 'P (后)' };
    };
    const labels = getLabels();

    if (!volumeData) {
        return (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#ffffff', color: '#94a3b8', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                <span>LOADING VTK SLICE DATA...</span>
            </div>
        );
    }

    const axisColor = axis === 'axial' ? '#16a34a' : (axis === 'coronal' ? '#d97706' : '#e11d48');

    return (
        <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: '#ffffff' }}>
            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

            {/* 四周解剖方向指示芯片 (Pixel-Perfect Light Chip Style) */}
            <div style={{ position: 'absolute', top: 7, left: '50%', transform: 'translateX(-50%)', fontFamily: 'var(--font-mono)', fontSize: 9.5, color: axisColor, fontWeight: 700, pointerEvents: 'none', background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(6px)', padding: '2px 8px', borderRadius: 4, border: `1px solid ${axisColor}44`, boxShadow: '0 2px 6px rgba(0,0,0,0.06)' }}>
                {labels.top}
            </div>
            <div style={{ position: 'absolute', bottom: 7, left: '50%', transform: 'translateX(-50%)', fontFamily: 'var(--font-mono)', fontSize: 9.5, color: axisColor, fontWeight: 700, pointerEvents: 'none', background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(6px)', padding: '2px 8px', borderRadius: 4, border: `1px solid ${axisColor}44`, boxShadow: '0 2px 6px rgba(0,0,0,0.06)' }}>
                {labels.bottom}
            </div>
            <div style={{ position: 'absolute', left: 7, top: '50%', transform: 'translateY(-50%)', fontFamily: 'var(--font-mono)', fontSize: 9.5, color: axisColor, fontWeight: 700, pointerEvents: 'none', background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(6px)', padding: '2px 8px', borderRadius: 4, border: `1px solid ${axisColor}44`, boxShadow: '0 2px 6px rgba(0,0,0,0.06)' }}>
                {labels.left}
            </div>
            <div style={{ position: 'absolute', right: 7, top: '50%', transform: 'translateY(-50%)', fontFamily: 'var(--font-mono)', fontSize: 9.5, color: axisColor, fontWeight: 700, pointerEvents: 'none', background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(6px)', padding: '2px 8px', borderRadius: 4, border: `1px solid ${axisColor}44`, boxShadow: '0 2px 6px rgba(0,0,0,0.06)' }}>
                {labels.right}
            </div>
        </div>
    );
}
