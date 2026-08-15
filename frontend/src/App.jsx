import React, { useState, useEffect, useRef } from 'react';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import Vtk2DSliceViewer from './components/Vtk2DSliceViewer';
import Vtk3DVolumeViewer from './components/Vtk3DVolumeViewer';
import ErrorBoundary from './components/ErrorBoundary';

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
    imageData.setDimensions(payload.dimensions || [182, 218, 182]);
    imageData.setSpacing(payload.spacing || [1.0, 1.0, 1.0]);
    imageData.setOrigin(payload.origin || [0.0, 0.0, 0.0]);

    const scalars = vtkDataArray.newInstance({
        name: 'Scalars',
        numberOfComponents: 1,
        values: bytes
    });
    imageData.getPointData().setScalars(scalars);
    return imageData;
}

export default function App() {
    // 切片索引 (Axial 182, Coronal 218, Sagittal 182)
    const [axialIndex, setAxialIndex] = useState(90);
    const [coronalIndex, setCoronalIndex] = useState(109);
    const [sagittalIndex, setSagittalIndex] = useState(91);

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
            id: 1,
            sender: 'agent',
            text: '您好！RadPilot 影像工作站已全面升级为【纯原生 @kitware/vtk.js 架构】。正交三视图与 3D 体渲染视口全部由 VTK.js 原生驱动，支持 WW/WL 实时调谐与独立子 Renderer 坐标轴系统。',
            meta: { source: 'gemini_api' }
        }
    ]);
    const [isProcessing, setIsProcessing] = useState(false);
    const chatBottomRef = useRef(null);

    // 1. 初始化拉取系统信息与 VTK 主体数据
    useEffect(() => {
        fetch('/api/info')
            .then(res => {
                if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
                return res.json();
            })
            .then(data => {
                setHarnessState(data.harness_state || 'PAUSED_FOR_DOCTOR');
                setCurrentVersion(data.current_version || 'v0');
                setLoadError(null);
            })
            .catch(err => {
                console.error('获取系统信息失败:', err);
                setLoadError('后端服务连接中断，尝试重连中...');
            });

        // 加载主 MRI vtkImageData
        fetch('/api/volume_data_vtk')
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.json();
            })
            .then(payload => {
                const img = buildVtkImageDataFromPayload(payload);
                setMriImageData(img);
            })
            .catch(err => console.error('加载 MRI 体数据失败:', err));
    }, []);

    // 2. 当 Mask 版本更新时，拉取最新的 Mask vtkImageData
    useEffect(() => {
        fetch('/api/mask_volume_vtk')
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.json();
            })
            .then(payload => {
                if (payload && payload.has_mask) {
                    const maskImg = buildVtkImageDataFromPayload(payload);
                    setMaskImageData(maskImg);
                } else {
                    setMaskImageData(null);
                }
            })
            .catch(err => console.error('加载 Mask 体数据失败:', err));
    }, [currentVersion]);

    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages]);

    // 当前可见的分割图层
    const activeLayer = labelLayers.find(l => l.visible) || null;

    const toggleLayerVisible = (id) => {
        setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, visible: !l.visible } : l));
    };

    const changeLayerColor = (id, color) => {
        setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, color } : l));
    };

    const changeLayerOpacity = (id, opacity) => {
        setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, opacity: Number(opacity) } : l));
    };

    // 自然语言交互驱动 Gemini API
    const handleSendMessage = () => {
        if (!inputText.trim() || isProcessing) return;
        const msgText = inputText.trim();
        setInputText('');

        setChatMessages(prev => [...prev, { id: Date.now(), sender: 'user', text: msgText }]);
        setIsProcessing(true);
        setHarnessState('PROCESSING');

        fetch('/api/interact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: msgText })
        })
            .then(res => res.json())
            .then(data => {
                setIsProcessing(false);
                setChatMessages(prev => [...prev, {
                    id: Date.now() + 1,
                    sender: 'agent',
                    text: data.message || data.explanation,
                    meta: { action: data.action, source: data.source, elapsed: data.elapsed_ms }
                }]);
                setHarnessState(data.state || 'PAUSED_FOR_DOCTOR');
                setCurrentVersion(data.current_version);

                if (['SKULL_STRIP', 'EXPAND', 'SHRINK', 'REMOVE_ARTIFACTS'].includes(data.action)) {
                    const versionId = data.current_version;
                    let layerName = `${versionId}: 脑实质标注`;
                    let defaultColor = '#06b6d4';

                    if (msgText.includes('左')) {
                        layerName = `${versionId}: 左半脑实质 (Left)`;
                        defaultColor = '#3b82f6';
                    } else if (msgText.includes('右')) {
                        layerName = `${versionId}: 右半脑实质 (Right)`;
                        defaultColor = '#ef4444';
                    } else if (data.action === 'SKULL_STRIP') {
                        layerName = `${versionId}: 全脑实质 (Full Brain)`;
                        defaultColor = '#06b6d4';
                    }

                    setLabelLayers(prev => [
                        { id: versionId, name: layerName, visible: true, color: defaultColor, opacity: 0.6 },
                        ...prev.filter(l => l.id !== versionId)
                    ]);
                }
            })
            .catch(err => {
                console.error('交互错误:', err);
                setIsProcessing(false);
                setHarnessState('PAUSED_FOR_DOCTOR');
            });
    };

    return (
        <div className="radpilot-app">
            {/* 1. 顶部 Header */}
            <header className="app-header">
                <div className="brand">
                    <span className="brand-icon">🧠</span>
                    <span className="brand-title">RadPilot VTK PACS Workstation</span>
                </div>
                <div className="header-status">
                    <div className="status-pill">
                        <span className={`dot-indicator ${harnessState.toLowerCase()}`}></span>
                        <span>
                            {harnessState === 'PROCESSING' && 'Gemini API 分析中...'}
                            {harnessState === 'PAUSED_FOR_DOCTOR' && '等待医生在环指令'}
                            {harnessState === 'COMPLETED' && '金标导出完成'}
                        </span>
                    </div>
                    <div className="status-pill">
                        <span style={{ color: '#06b6d4', fontWeight: 600 }}>Mask 版本: {currentVersion}</span>
                    </div>
                </div>
            </header>

            {/* 2. 主体三栏 */}
            <div className="app-body-triplanar">

                {/* 【左侧栏】：窗宽窗位 + 3D ISO 阈值 + 图层管理器 */}
                <aside className="layers-sidebar">
                    <div className="sidebar-title">PACS 调控与图层管理</div>

                    {/* WW / WL 面板 */}
                    <div className="panel-section">
                        <div className="section-label">
                            <span>DICOM 窗宽窗位 (WW / WL)</span>
                        </div>
                        <div className="layer-card">
                            <div className="control-row" style={{ marginBottom: 6 }}>
                                <span>窗宽 (WW): <b style={{ color: '#38bdf8' }}>{windowWidth}</b></span>
                                <input
                                    type="range" min="500" max="10000" step="100"
                                    value={windowWidth}
                                    onChange={(e) => setWindowWidth(Number(e.target.value))}
                                    style={{ width: 105 }}
                                />
                            </div>
                            <div className="control-row" style={{ marginBottom: 8 }}>
                                <span>窗位 (WL): <b style={{ color: '#38bdf8' }}>{windowLevel}</b></span>
                                <input
                                    type="range" min="0" max="8000" step="100"
                                    value={windowLevel}
                                    onChange={(e) => setWindowLevel(Number(e.target.value))}
                                    style={{ width: 105 }}
                                />
                            </div>
                            <div style={{ display: 'flex', gap: 6 }}>
                                <button
                                    style={{ flex: 1, padding: '4px', background: '#1e293b', border: '1px solid #334155', color: '#f8fafc', borderRadius: 4, fontSize: 10.5, cursor: 'pointer' }}
                                    onClick={() => { setWindowWidth(7000); setWindowLevel(3500); }}
                                >
                                    软组织
                                </button>
                                <button
                                    style={{ flex: 1, padding: '4px', background: '#1e293b', border: '1px solid #334155', color: '#f8fafc', borderRadius: 4, fontSize: 10.5, cursor: 'pointer' }}
                                    onClick={() => { setWindowWidth(4000); setWindowLevel(2500); }}
                                >
                                    高对比
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* 3D ISO 阈值面板 */}
                    <div className="panel-section">
                        <div className="section-label">
                            <span>3D VTK 体渲染 ISO 阈值</span>
                        </div>
                        <div className="layer-card">
                            <div className="control-row">
                                <span>3D 阈值: <b style={{ color: '#06b6d4' }}>{Math.round(volumeThreshold * 100)}%</b></span>
                                <input
                                    type="range" min="0.05" max="0.80" step="0.02"
                                    value={volumeThreshold}
                                    onChange={(e) => setVolumeThreshold(Number(e.target.value))}
                                    style={{ width: 105 }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* 主体数据 */}
                    <div className="panel-section">
                        <div className="section-label"><span>主体数据序列</span></div>
                        <div className="layer-card" style={{ marginBottom: 0 }}>
                            <div className="layer-header" style={{ marginBottom: 0 }}>
                                <span className="layer-title"><span>🧊 MNI152 T1w MRI</span></span>
                                <button className={`toggle-eye ${showVolume ? 'active' : ''}`} onClick={() => setShowVolume(!showVolume)}>
                                    {showVolume ? '👁️' : '🙈'}
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* 动态 Label 图层列表 */}
                    <div className="panel-section" style={{ flex: 1, borderBottom: 'none' }}>
                        <div className="section-label"><span>分割标签图层 (Label Layers)</span></div>

                        {labelLayers.length === 0 ? (
                            <div style={{ padding: '12px 8px', textAlign: 'center', color: '#64748b', fontSize: 11, border: '1px dashed #1e293b', borderRadius: 6 }}>
                                暂无分割图层<br/>输入“分割左脑”动态生成
                            </div>
                        ) : (
                            labelLayers.map(layer => (
                                <div key={layer.id} className="layer-card">
                                    <div className="layer-header">
                                        <span className="layer-title">
                                            <span style={{ color: layer.color }}>●</span>
                                            <span style={{ fontSize: 11.5 }}>{layer.name}</span>
                                        </span>
                                        <button className={`toggle-eye ${layer.visible ? 'active' : ''}`} onClick={() => toggleLayerVisible(layer.id)}>
                                            {layer.visible ? '👁️' : '🙈'}
                                        </button>
                                    </div>
                                    {layer.visible && (
                                        <div className="layer-controls">
                                            <div className="control-row">
                                                <span>颜色:</span>
                                                <div className="color-swatch-list">
                                                    {['#06b6d4', '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'].map(c => (
                                                        <button key={c} className={`swatch-btn ${layer.color === c ? 'selected' : ''}`} style={{ backgroundColor: c }} onClick={() => changeLayerColor(layer.id, c)} />
                                                    ))}
                                                </div>
                                            </div>
                                            <div className="control-row">
                                                <span>透明度:</span>
                                                <input type="range" min="0.1" max="1.0" step="0.05" value={layer.opacity} onChange={(e) => changeLayerOpacity(layer.id, e.target.value)} style={{ width: 75 }} />
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
                    <div className="pane-view">
                        <div className="pane-header">
                            <span className="pane-badge badge-axial">VTK Z (轴位)</span>
                            <span>轴位 (XY 切面 / 上下向)</span>
                        </div>
                        <div className="pane-canvas-wrapper">
                            <ErrorBoundary>
                                <Vtk2DSliceViewer
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
                            <span style={{ fontSize: 10.5, color: '#22c55e', fontWeight: 700 }}>Z 轴位 (绿):</span>
                            <input type="range" min="0" max="181" value={axialIndex} onChange={(e) => setAxialIndex(Number(e.target.value))} className="pane-slider" />
                            <span className="pane-slice-text" style={{ color: '#22c55e' }}>{axialIndex} / 181</span>
                        </div>
                    </div>

                    {/* Pane 2: Coronal 冠状位 (Y 轴 - 黄色) */}
                    <div className="pane-view">
                        <div className="pane-header">
                            <span className="pane-badge badge-coronal">VTK Y (冠状)</span>
                            <span>冠状位 (XZ 切面 / 前后向)</span>
                        </div>
                        <div className="pane-canvas-wrapper">
                            <ErrorBoundary>
                                <Vtk2DSliceViewer
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
                            <span style={{ fontSize: 10.5, color: '#eab308', fontWeight: 700 }}>Y 冠状 (黄):</span>
                            <input type="range" min="0" max="217" value={coronalIndex} onChange={(e) => setCoronalIndex(Number(e.target.value))} className="pane-slider" />
                            <span className="pane-slice-text" style={{ color: '#eab308' }}>{coronalIndex} / 217</span>
                        </div>
                    </div>

                    {/* Pane 3: Sagittal 矢状位 (X 轴 - 红色) */}
                    <div className="pane-view">
                        <div className="pane-header">
                            <span className="pane-badge badge-sagittal">VTK X (矢状)</span>
                            <span>矢状位 (YZ 切面 / 左右向)</span>
                        </div>
                        <div className="pane-canvas-wrapper">
                            <ErrorBoundary>
                                <Vtk2DSliceViewer
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
                            <span style={{ fontSize: 10.5, color: '#ef4444', fontWeight: 700 }}>X 矢状 (红):</span>
                            <input type="range" min="0" max="181" value={sagittalIndex} onChange={(e) => setSagittalIndex(Number(e.target.value))} className="pane-slider" />
                            <span className="pane-slice-text" style={{ color: '#ef4444' }}>{sagittalIndex} / 181</span>
                        </div>
                    </div>

                    {/* Pane 4: 3D 原生 VTK 体渲染与独立子 Renderer 坐标轴 */}
                    <div className="pane-view">
                        <div className="pane-header">
                            <span className="pane-badge badge-3d">VTK 3D VOLUME</span>
                            <span>3D 体渲染 (GPU Ray Marching + 独立坐标轴)</span>
                        </div>
                        <div className="pane-canvas-wrapper" style={{ position: 'relative', width: '100%', height: '100%' }}>
                            <ErrorBoundary>
                                <Vtk3DVolumeViewer
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
                    <div className="sidebar-title">Gemini Vision Agent 对话</div>
                    <div className="chat-history">
                        {chatMessages.map(msg => (
                            <div key={msg.id} className={`chat-bubble ${msg.sender}`}>
                                <span className="bubble-sender">{msg.sender === 'user' ? '医生' : 'RadPilot Agent'}</span>
                                <div className="bubble-content">{msg.text}</div>
                                {msg.meta && (
                                    <div className="agent-meta">
                                        {msg.meta.action && <span className="meta-tag">Action: {msg.meta.action}</span>}
                                        {msg.meta.source && <span className="meta-tag">Source: {msg.meta.source}</span>}
                                        {msg.meta.elapsed !== undefined && <span className="meta-tag">{msg.meta.elapsed}ms</span>}
                                    </div>
                                )}
                            </div>
                        ))}
                        <div ref={chatBottomRef} />
                    </div>

                    <div className="chat-input-container">
                        <input
                            type="text" className="chat-input" placeholder="发送指令 (如: '分割左脑')..."
                            value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                        />
                        <button className="send-btn" onClick={handleSendMessage} disabled={isProcessing}>
                            {isProcessing ? '分析中...' : '发送'}
                        </button>
                    </div>
                </aside>

            </div>
        </div>
    );
}
