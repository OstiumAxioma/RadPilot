import React, { useState, useEffect, useRef } from 'react';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import { 
    Activity, 
    Brain, 
    Eye, 
    EyeOff, 
    Layers, 
    Sliders, 
    Send, 
    Sparkles, 
    Box, 
    CheckCircle2, 
    AlertCircle,
    Cpu,
    Flame,
    UploadCloud,
    FolderOpen
} from 'lucide-react';
import Vtk2DSliceViewer from './components/Vtk2DSliceViewer';
import Vtk3DVolumeViewer from './components/Vtk3DVolumeViewer';
import ErrorBoundary from './components/ErrorBoundary';
import MarkdownRenderer from './components/MarkdownRenderer';

/**
 * 将后端返回的标准 VTK Payload 转换为前端原生的 vtkImageData 实体
 */
function buildVtkImageDataFromPayload(payload) {
    if (!payload || !payload.raw_base64) return null;
    const binaryString = window.atob(payload.raw_base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    const imageData = vtkImageData.newInstance();
    imageData.setDimensions(payload.dimensions);
    imageData.setSpacing(payload.spacing || [1, 1, 1]);
    imageData.setOrigin(payload.origin || [0, 0, 0]);

    const scalars = vtkDataArray.newInstance({
        name: 'Scalars',
        numberOfComponents: 1,
        values: bytes
    });
    imageData.getPointData().setScalars(scalars);
    return imageData;
}

export default function App() {
    // 切片索引与最大边界 (自适应不同 NIfTI 尺寸)
    const [axialIndex, setAxialIndex] = useState(90);
    const [coronalIndex, setCoronalIndex] = useState(109);
    const [sagittalIndex, setSagittalIndex] = useState(91);
    const [maxSlices, setMaxSlices] = useState({ axial: 181, coronal: 217, sagittal: 181 });

    // 当前载入的序列名称与上传状态
    const [seriesName, setSeriesName] = useState('MNI152NLin6_res-1x1x1_T1w.nii');
    const [isUploading, setIsUploading] = useState(false);
    const fileInputRef = useRef(null);

    // 窗宽 (Window Width) 与 窗位 (Window Level)
    const [windowWidth, setWindowWidth] = useState(7000);
    const [windowLevel, setWindowLevel] = useState(3500);

    // 3D vtkVolume ISO 阈值
    const [volumeThreshold, setVolumeThreshold] = useState(0.20);

    // VTK ImageData 实体
    const [mriImageData, setMriImageData] = useState(null);
    const [maskImageData, setMaskImageData] = useState(null);

    const [harnessState, setHarnessState] = useState('PAUSED_FOR_DOCTOR');
    const [currentVersion, setCurrentVersion] = useState('v0');
    const [loadError, setLoadError] = useState(null);

    // 体数据显隐与动态 Label 图层
    const [showVolume, setShowVolume] = useState(true);
    const [labelLayers, setLabelLayers] = useState([]);

    // 自然语言对话
    const [inputText, setInputText] = useState('');
    const [chatMessages, setChatMessages] = useState([
        {
            id: 'init-1',
            sender: 'agent',
            text: 'RadPilot VTK.js 影像工作站已就绪。默认载入 MNI152 标准脑核磁序列，支持上传新 NIfTI / DICOM 影像。请输入自然语言指令进行交互。',
            meta: { action: 'READY', source: 'VTK_ENGINE' }
        }
    ]);
    const [isProcessing, setIsProcessing] = useState(false);
    const chatBottomRef = useRef(null);

    // 序列唯一标识 (用于强制销毁并重建 VTK 视口管线，防止旧影像/旧纹理残留)
    const [seriesId, setSeriesId] = useState('default_mni152');

    // 封装拉取 MRI 主体数据与信息 (彻底重置场景，只保留单一最新影像)
    const loadMriData = (newSeriesId = null) => {
        // 1. 立即清空旧数据引用与旧分割，防止多影像或多图层并存
        setMriImageData(null);
        setMaskImageData(null);
        setLabelLayers([]);
        setCurrentVersion('v0');

        const timestamp = Date.now();
        fetch(`http://localhost:8000/api/info?t=${timestamp}`)
            .then(res => res.json())
            .then(info => {
                if (info.image_name) setSeriesName(info.image_name);
                if (info.slices_info) {
                    const si = info.slices_info;
                    setMaxSlices({
                        axial: si.axial_slices - 1,
                        coronal: si.coronal_slices - 1,
                        sagittal: si.sagittal_slices - 1
                    });
                    setAxialIndex(Math.floor(si.axial_slices / 2));
                    setCoronalIndex(Math.floor(si.coronal_slices / 2));
                    setSagittalIndex(Math.floor(si.sagittal_slices / 2));
                }
            })
            .catch(err => console.warn('获取图像 info 失败:', err));

        fetch(`http://localhost:8000/api/volume_data_vtk?t=${timestamp}`)
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then(data => {
                console.log('成功获取 MRI VTK Payload, 大小:', data.dimensions);
                const img = buildVtkImageDataFromPayload(data);
                setMriImageData(img);
                setSeriesId(newSeriesId || `series_${timestamp}`);
            })
            .catch(err => {
                console.error('拉取 VTK MRI 数据失败:', err);
                setLoadError(`拉取 MRI 数据失败: ${err.message}`);
            });
    };

    // 1. 初始化拉取默认 MRI 主体数据
    useEffect(() => {
        loadMriData('default_mni152');
    }, []);

    // 2. 监听当前 Mask 版本拉取 VTK Mask 数据
    useEffect(() => {
        if (!currentVersion || currentVersion === 'v0') {
            setMaskImageData(null);
            return;
        }

        fetch(`http://localhost:8000/api/mask_volume_vtk?version=${currentVersion}&t=${Date.now()}`)
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then(data => {
                console.log('成功获取 Mask VTK Payload, 版本:', currentVersion);
                const img = buildVtkImageDataFromPayload(data);
                setMaskImageData(img);
            })
            .catch(err => {
                console.error('拉取 VTK Mask 数据失败:', err);
            });
    }, [currentVersion]);

    // 处理新影像文件上传 (自动刷新 + 彻底清除旧影像与分割)
    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        fetch('http://localhost:8000/api/upload_image', {
            method: 'POST',
            body: formData
        })
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then(data => {
                setIsUploading(false);
                const newSeriesName = data.image_name || file.name;
                const newSeriesId = data.series_id || `series_${Date.now()}`;
                setSeriesName(newSeriesName);
                
                // 彻底刷新载入新数据并重置视口
                loadMriData(newSeriesId);

                // 对话栏明确提示场景已重置
                setChatMessages([
                    {
                        id: Date.now().toString(),
                        sender: 'agent',
                        text: `【场景已彻底重置】已清除旧影像与历史分割。当前场景仅保留单一最新影像: ${newSeriesName}。所有后续 AI 分析与分割指令将仅针对本序列执行。`,
                        meta: { action: 'RESET_AND_LOAD', source: 'WORKSPACE_ENGINE' }
                    }
                ]);
            })
            .catch(err => {
                console.error('上传失败:', err);
                setIsUploading(false);
                alert(`上传影像失败: ${err.message}`);
            });
    };

    // 滚动对话到底部
    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages]);

    // 图层操作
    const toggleLayerVisible = (id) => {
        setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, visible: !l.visible } : l));
    };

    const changeLayerColor = (id, newColor) => {
        setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, color: newColor } : l));
    };

    const changeLayerOpacity = (id, newOpacity) => {
        setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, opacity: parseFloat(newOpacity) } : l));
    };

    const activeLayer = labelLayers.find(l => l.id === currentVersion) || {
        visible: true,
        color: '#06b6d4',
        opacity: 0.6
    };

    // 发送自然语言对话指令
    const handleSendMessage = () => {
        if (!inputText.trim() || isProcessing) return;

        const userMsg = {
            id: Date.now().toString(),
            sender: 'user',
            text: inputText.trim()
        };
        setChatMessages(prev => [...prev, userMsg]);
        setInputText('');
        setIsProcessing(true);
        setHarnessState('PROCESSING');

        fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: userMsg.text, message: userMsg.text, current_version: currentVersion })
        })
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then(data => {
                setIsProcessing(false);
                const replyText = data.reply || data.message || '指令执行完毕。';
                
                setChatMessages(prev => [
                    ...prev,
                    {
                        id: (Date.now() + 1).toString(),
                        sender: 'agent',
                        text: replyText,
                        meta: {
                            action: data.action_type || data.action || 'INSPECT',
                            source: data.source || 'GEMINI_ROUTER',
                            elapsed: data.elapsed_ms || 0
                        }
                    }
                ]);

                const versionId = data.new_version || data.current_version;
                if (versionId && versionId !== 'v0') {
                    setCurrentVersion(versionId);

                    const colorPalette = ['#06b6d4', '#3b82f6', '#e11d48', '#16a34a', '#d97706', '#9333ea'];
                    const defaultColor = colorPalette[labelLayers.length % colorPalette.length];
                    const layerName = data.layer_name || `分割图层 (${versionId})`;

                    setLabelLayers(prev => [
                        { id: versionId, name: layerName, visible: true, color: defaultColor, opacity: 0.6 },
                        ...prev.filter(l => l.id !== versionId)
                    ]);
                }
                setHarnessState(data.state || 'PAUSED_FOR_DOCTOR');
            })
            .catch(err => {
                console.error('交互错误:', err);
                setIsProcessing(false);
                setChatMessages(prev => [
                    ...prev,
                    {
                        id: (Date.now() + 1).toString(),
                        sender: 'agent',
                        text: `通信异常: 无法连接至后端服务 (${err.message})`,
                        meta: { action: 'ERROR', source: 'NETWORK' }
                    }
                ]);
                setHarnessState('PAUSED_FOR_DOCTOR');
            });
    };

    return (
        <div className="radpilot-app">
            {/* 1. 顶部 Header (Pixel-Perfect 精密导航栏) */}
            <header className="app-header">
                <div className="brand">
                    <div className="brand-icon-wrapper">
                        <Brain size={15} />
                    </div>
                    <div className="brand-title">
                        <span>RadPilot</span>
                        <span className="brand-tag">VTK.JS PACS</span>
                    </div>
                </div>
                <div className="header-status">
                    <div className="status-pill">
                        <span className={`dot-indicator ${harnessState.toLowerCase()}`}></span>
                        <span>
                            {harnessState === 'PROCESSING' && 'Gemini API 分析中...'}
                            {harnessState === 'PAUSED_FOR_DOCTOR' && '等待医生指令'}
                            {harnessState === 'COMPLETED' && '金标导出完成'}
                        </span>
                    </div>
                    <div className="status-pill" style={{ borderColor: 'rgba(2, 132, 199, 0.3)' }}>
                        <span style={{ color: '#0284c7', fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: 10.5 }}>MASK: {currentVersion}</span>
                    </div>
                </div>
            </header>

            {/* 2. 主体三栏 */}
            <div className="app-body-triplanar">

                {/* 【左侧栏】：窗宽窗位 + 3D ISO 阈值 + 图层管理器 */}
                <aside className="layers-sidebar">
                    <div className="sidebar-title">
                        <span>PACS 调控与图层</span>
                        <span style={{ fontSize: 9.5, opacity: 0.5, fontFamily: 'var(--font-mono)' }}>v1.0.0</span>
                    </div>

                    {/* WW / WL 面板 */}
                    <div className="panel-section">
                        <div className="section-label">
                            <Sliders size={12} />
                            <span>DICOM 窗宽窗位 (WW / WL)</span>
                        </div>
                        <div className="pp-corner-box layer-card">
                            <div className="control-row" style={{ marginBottom: 8 }}>
                                <span>窗宽 (WW): <b style={{ color: '#0284c7' }}>{windowWidth}</b></span>
                                <input
                                    type="range" min="500" max="10000" step="100"
                                    value={windowWidth}
                                    onChange={(e) => setWindowWidth(Number(e.target.value))}
                                    className="pixel-slider"
                                    style={{ width: 110 }}
                                />
                            </div>
                            <div className="control-row" style={{ marginBottom: 10 }}>
                                <span>窗位 (WL): <b style={{ color: '#0284c7' }}>{windowLevel}</b></span>
                                <input
                                    type="range" min="0" max="8000" step="100"
                                    value={windowLevel}
                                    onChange={(e) => setWindowLevel(Number(e.target.value))}
                                    className="pixel-slider"
                                    style={{ width: 110 }}
                                />
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                                <button
                                    className="pp-btn"
                                    onClick={() => { setWindowWidth(7000); setWindowLevel(3500); }}
                                >
                                    软组织预设
                                </button>
                                <button
                                    className="pp-btn"
                                    onClick={() => { setWindowWidth(4000); setWindowLevel(2500); }}
                                >
                                    高对比预设
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* 3D ISO 阈值面板 */}
                    <div className="panel-section">
                        <div className="section-label">
                            <Box size={12} />
                            <span>3D 体渲染 ISO 阈值</span>
                        </div>
                        <div className="pp-corner-box layer-card">
                            <div className="control-row">
                                <span>3D 阈值: <b style={{ color: '#9333ea' }}>{Math.round(volumeThreshold * 100)}%</b></span>
                                <input
                                    type="range" min="0.05" max="0.80" step="0.02"
                                    value={volumeThreshold}
                                    onChange={(e) => setVolumeThreshold(Number(e.target.value))}
                                    className="pixel-slider"
                                    style={{ width: 110 }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* 主体数据序列与上传新影像入口 */}
                    <div className="panel-section">
                        <div className="section-label">
                            <Layers size={12} />
                            <span>主体数据序列</span>
                        </div>
                        <div className="pp-corner-box layer-card" style={{ marginBottom: 0 }}>
                            <div className="layer-header" style={{ marginBottom: 0 }}>
                                <span className="layer-title" title={seriesName} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 190 }}>
                                    <Cpu size={12} style={{ color: '#0284c7', flexShrink: 0 }} />
                                    <span style={{ fontSize: 11 }}>{seriesName}</span>
                                </span>
                                <button className="pp-btn-icon" onClick={() => setShowVolume(!showVolume)} title="显隐切换">
                                    {showVolume ? <Eye size={13} style={{ color: '#0284c7' }} /> : <EyeOff size={13} style={{ color: '#94a3b8' }} />}
                                </button>
                            </div>
                        </div>

                        {/* 上传新序列按钮与隐藏 Input */}
                        <div style={{ marginTop: 6 }}>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".nii,.nii.gz,.dcm"
                                style={{ display: 'none' }}
                                onChange={handleFileChange}
                            />
                            <button
                                className="pp-btn"
                                style={{ width: '100%', fontSize: 10.5, padding: '5px 8px', gap: 6, color: '#0284c7' }}
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isUploading}
                            >
                                <UploadCloud size={13} />
                                <span>{isUploading ? '正在解析重构新影像...' : '载入新影像 (NIfTI / DICOM)'}</span>
                            </button>
                        </div>
                    </div>

                    {/* 动态 Label 图层列表 */}
                    <div className="panel-section" style={{ flex: 1, borderBottom: 'none' }}>
                        <div className="section-label">
                            <Sparkles size={12} />
                            <span>分割标签图层 (Label Layers)</span>
                        </div>

                        {labelLayers.length === 0 ? (
                            <div className="pp-dashed-border" style={{ padding: '14px 8px', textAlign: 'center', color: '#94a3b8', fontSize: 11, borderRadius: 4, background: '#f8fafc' }}>
                                暂无分割图层<br/>输入“分割左脑”自动生成
                            </div>
                        ) : (
                            labelLayers.map(layer => (
                                <div key={layer.id} className="pp-corner-box layer-card" style={{ marginBottom: 6 }}>
                                    <div className="layer-header">
                                        <span className="layer-title">
                                            <span style={{ width: 8, height: 8, borderRadius: 2, backgroundColor: layer.color, display: 'inline-block' }}></span>
                                            <span style={{ fontSize: 11.5 }}>{layer.name}</span>
                                        </span>
                                        <button className="pp-btn-icon" onClick={() => toggleLayerVisible(layer.id)}>
                                            {layer.visible ? <Eye size={13} style={{ color: layer.color }} /> : <EyeOff size={13} style={{ color: '#94a3b8' }} />}
                                        </button>
                                    </div>
                                    {layer.visible && (
                                        <div style={{ marginTop: 8, paddingTop: 6, borderTop: '1px solid var(--border-subtle)' }}>
                                            <div className="control-row" style={{ marginBottom: 6 }}>
                                                <span>颜色:</span>
                                                <div className="color-swatch-list">
                                                    {['#06b6d4', '#3b82f6', '#e11d48', '#16a34a', '#d97706', '#9333ea'].map(c => (
                                                        <button key={c} className={`swatch-btn ${layer.color === c ? 'selected' : ''}`} style={{ backgroundColor: c }} onClick={() => changeLayerColor(layer.id, c)} />
                                                    ))}
                                                </div>
                                            </div>
                                            <div className="control-row">
                                                <span>透明度:</span>
                                                <input type="range" min="0.1" max="1.0" step="0.05" value={layer.opacity} onChange={(e) => changeLayerOpacity(layer.id, e.target.value)} className="pixel-slider" style={{ width: 85 }} />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))
                        )}
                    </div>
                </aside>

                {/* 【中间 4 宫格】：正交三视图 + 3D 体渲染全 VTK.js 原生视口 */}
                <section className="viewport-grid">
                    {/* Pane 1: Axial 轴位 (Z 轴 - 绿色) */}
                    <div className="pp-corner-box pane-view">
                        <div className="pane-header">
                            <span className="pane-badge badge-axial">VTK Z (轴位)</span>
                            <span>轴状位 (Axial MPR)</span>
                        </div>
                        <div className="pane-canvas-wrapper">
                            <ErrorBoundary>
                                <Vtk2DSliceViewer
                                    key={`axial_${seriesId}`}
                                    axis="axial"
                                    sliceIndex={axialIndex}
                                    volumeData={mriImageData}
                                    maskData={maskImageData}
                                    windowWidth={windowWidth}
                                    windowLevel={windowLevel}
                                    activeLayer={activeLayer}
                                />
                            </ErrorBoundary>
                        </div>
                        <div className="pane-slider-bar">
                            <span style={{ fontSize: 9.5, color: '#16a34a', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>Z 轴位:</span>
                            <input type="range" min="0" max={maxSlices.axial} value={axialIndex} onChange={(e) => setAxialIndex(Number(e.target.value))} className="pane-slider pixel-slider" />
                            <span className="pane-slice-text" style={{ color: '#16a34a' }}>{axialIndex} / {maxSlices.axial}</span>
                        </div>
                    </div>

                    {/* Pane 2: Coronal 冠状位 (Y 轴 - 黄色) */}
                    <div className="pp-corner-box pane-view">
                        <div className="pane-header">
                            <span className="pane-badge badge-coronal">VTK Y (冠状)</span>
                            <span>冠状位 (Coronal MPR)</span>
                        </div>
                        <div className="pane-canvas-wrapper">
                            <ErrorBoundary>
                                <Vtk2DSliceViewer
                                    key={`coronal_${seriesId}`}
                                    axis="coronal"
                                    sliceIndex={coronalIndex}
                                    volumeData={mriImageData}
                                    maskData={maskImageData}
                                    windowWidth={windowWidth}
                                    windowLevel={windowLevel}
                                    activeLayer={activeLayer}
                                />
                            </ErrorBoundary>
                        </div>
                        <div className="pane-slider-bar">
                            <span style={{ fontSize: 9.5, color: '#d97706', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>Y 冠状:</span>
                            <input type="range" min="0" max={maxSlices.coronal} value={coronalIndex} onChange={(e) => setCoronalIndex(Number(e.target.value))} className="pane-slider pixel-slider" />
                            <span className="pane-slice-text" style={{ color: '#d97706' }}>{coronalIndex} / {maxSlices.coronal}</span>
                        </div>
                    </div>

                    {/* Pane 3: Sagittal 矢状位 (X 轴 - 红色) */}
                    <div className="pp-corner-box pane-view">
                        <div className="pane-header">
                            <span className="pane-badge badge-sagittal">VTK X (矢状)</span>
                            <span>矢状位 (Sagittal MPR)</span>
                        </div>
                        <div className="pane-canvas-wrapper">
                            <ErrorBoundary>
                                <Vtk2DSliceViewer
                                    key={`sagittal_${seriesId}`}
                                    axis="sagittal"
                                    sliceIndex={sagittalIndex}
                                    volumeData={mriImageData}
                                    maskData={maskImageData}
                                    windowWidth={windowWidth}
                                    windowLevel={windowLevel}
                                    activeLayer={activeLayer}
                                />
                            </ErrorBoundary>
                        </div>
                        <div className="pane-slider-bar">
                            <span style={{ fontSize: 9.5, color: '#e11d48', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>X 矢状:</span>
                            <input type="range" min="0" max={maxSlices.sagittal} value={sagittalIndex} onChange={(e) => setSagittalIndex(Number(e.target.value))} className="pane-slider pixel-slider" />
                            <span className="pane-slice-text" style={{ color: '#e11d48' }}>{sagittalIndex} / {maxSlices.sagittal}</span>
                        </div>
                    </div>

                    {/* Pane 4: 3D 原生 VTK 体渲染与独立子 Renderer 坐标轴 */}
                    <div className="pp-corner-box pane-view">
                        <div className="pane-header">
                            <span className="pane-badge badge-3d">VTK 3D VOLUME</span>
                            <span>3D 容积渲染 (Volume Rendering)</span>
                        </div>
                        <div className="pane-canvas-wrapper" style={{ position: 'relative', width: '100%', height: '100%' }}>
                            <ErrorBoundary>
                                <Vtk3DVolumeViewer
                                    key={`3d_${seriesId}`}
                                    volumeData={mriImageData}
                                    maskData={maskImageData}
                                    windowWidth={windowWidth}
                                    windowLevel={windowLevel}
                                    volumeThreshold={volumeThreshold}
                                    showVolume={showVolume}
                                    activeLayer={activeLayer}
                                />
                            </ErrorBoundary>
                        </div>
                    </div>
                </section>

                {/* 【右侧栏】：Gemini Agent 对话区 */}
                <aside className="agent-sidebar">
                    <div className="sidebar-title">
                        <span>Gemini Vision Agent</span>
                        <Sparkles size={12} style={{ color: '#0284c7' }} />
                    </div>
                    <div className="chat-history">
                        {chatMessages.map(msg => (
                            <div key={msg.id} className={`chat-bubble ${msg.sender}`}>
                                <span className="bubble-sender">
                                    {msg.sender === 'user' ? '医生' : 'RadPilot Agent'}
                                </span>
                                <div className="bubble-content">
                                    <MarkdownRenderer content={msg.text} />
                                </div>
                                {msg.meta && (
                                    <div className="agent-meta">
                                        {msg.meta.action && <span className="meta-tag">ACTION: {msg.meta.action}</span>}
                                        {msg.meta.source && <span className="meta-tag">SOURCE: {msg.meta.source}</span>}
                                        {msg.meta.elapsed !== undefined && <span className="meta-tag">{msg.meta.elapsed}ms</span>}
                                    </div>
                                )}
                            </div>
                        ))}
                        {isProcessing && (
                            <div className="chat-bubble agent" style={{ borderLeftColor: '#0284c7', background: '#f8fafc' }}>
                                <span className="bubble-sender">RadPilot Agent</span>
                                <div className="bubble-content" style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#0284c7', fontWeight: 600 }}>
                                    <Sparkles size={12} />
                                    <span>Gemini 意图解析与医学算子调度中...</span>
                                </div>
                            </div>
                        )}
                        <div ref={chatBottomRef} />
                    </div>

                    <div className="chat-input-container">
                        <input
                            type="text" className="chat-input" placeholder="输入自然语言指令 (如: '分割左脑实质')..."
                            value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                        />
                        <button className="pp-btn pp-btn-primary" onClick={handleSendMessage} disabled={isProcessing} style={{ padding: '0 12px', height: '32px' }}>
                            <Send size={12} />
                            <span>{isProcessing ? '分析中' : '发送'}</span>
                        </button>
                    </div>
                </aside>

            </div>
        </div>
    );
}
