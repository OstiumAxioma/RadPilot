import React, { useState, useEffect, useRef } from 'react';
import Real3DViewer from './components/Real3DViewer';
import ErrorBoundary from './components/ErrorBoundary';

export default function App() {
  // 切片索引 (Axial 182, Coronal 218, Sagittal 182)
  const [axialIndex, setAxialIndex] = useState(90);
  const [coronalIndex, setCoronalIndex] = useState(109);
  const [sagittalIndex, setSagittalIndex] = useState(91);

  // 窗宽 (Window Width) 与 窗位 (Window Level) - 完美匹配 MNI152 NIfTI 的 0-9999 灰度标量量程
  const [windowWidth, setWindowWidth] = useState(7000);
  const [windowLevel, setWindowLevel] = useState(3500);

  // 3D vtkVolume ISO 阈值
  const [volumeThreshold, setVolumeThreshold] = useState(0.20);

  // 三视图 Base64 数据 Bundle
  const [triplanarData, setTriplanarData] = useState(null);
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
      text: '您好！RadPilot PACS 工作站已升级为【MNI152 0-9999 精准标量量程】。您可以在左侧面板自由调节窗宽(WW: 7000)与窗位(WL: 3500)，体验无死白的医学三视图与 VTK.js 体渲染。',
      meta: { source: 'gemini_api' }
    }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Canvas Refs
  const axialMriRef = useRef(null); const axialMaskRef = useRef(null);
  const coronalMriRef = useRef(null); const coronalMaskRef = useRef(null);
  const sagittalMriRef = useRef(null); const sagittalMaskRef = useRef(null);
  const chatBottomRef = useRef(null);

  // 1. 初始化拉取系统信息
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
  }, []);

  // 2. 当切片或 WW/WL 改变时，拉取最新三视图 Base64 数据
  useEffect(() => {
    fetchTriplanarData(axialIndex, coronalIndex, sagittalIndex, windowWidth, windowLevel);
  }, [axialIndex, coronalIndex, sagittalIndex, currentVersion, windowWidth, windowLevel]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const fetchTriplanarData = (ax, co, sa, ww, wl) => {
    fetch(`/api/triplanar?axial=${ax}&coronal=${co}&sagittal=${sa}&ww=${ww}&wl=${wl}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
        return res.json();
      })
      .then(data => {
        setTriplanarData(data);
        setHarnessState(data.harness_state);
        setCurrentVersion(data.current_version);
        setLoadError(null);
      })
      .catch(err => console.error('加载三视图失败:', err));
  };

  // 3. 渲染 2D Viewport Canvas
  useEffect(() => {
    if (!triplanarData) return;

    const render2DPane = (mriCanvas, maskCanvas, bundleKey) => {
      if (!mriCanvas || !maskCanvas || !triplanarData[bundleKey]) return;
      const ctx = mriCanvas.getContext('2d');
      const maskCtx = maskCanvas.getContext('2d');

      const mriImg = new Image();
      mriImg.onload = () => {
        mriCanvas.width = mriImg.width; mriCanvas.height = mriImg.height;
        maskCanvas.width = mriImg.width; maskCanvas.height = mriImg.height;

        ctx.clearRect(0, 0, mriImg.width, mriImg.height);
        ctx.drawImage(mriImg, 0, 0);

        maskCtx.clearRect(0, 0, mriImg.width, mriImg.height);
        const activeLayer = labelLayers.find(l => l.visible);

        if (activeLayer && triplanarData[bundleKey].mask_base64) {
          const maskImg = new Image();
          maskImg.onload = () => {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = mriImg.width; tempCanvas.height = mriImg.height;
            const tempCtx = tempCanvas.getContext('2d');

            tempCtx.drawImage(maskImg, 0, 0);
            const imgData = tempCtx.getImageData(0, 0, mriImg.width, mriImg.height);
            const data = imgData.data;

            const rgb = hexToRgb(activeLayer.color);
            for (let i = 0; i < data.length; i += 4) {
              if (data[i] > 20) {
                data[i] = rgb.r;
                data[i + 1] = rgb.g;
                data[i + 2] = rgb.b;
                data[i + 3] = Math.floor(255 * activeLayer.opacity);
              } else {
                data[i + 3] = 0;
              }
            }
            tempCtx.putImageData(imgData, 0, 0);
            maskCtx.drawImage(tempCanvas, 0, 0);
          };
          maskImg.src = triplanarData[bundleKey].mask_base64;
        }
      };
      mriImg.src = triplanarData[bundleKey].image_base64;
    };

    render2DPane(axialMriRef.current, axialMaskRef.current, 'axial');
    render2DPane(coronalMriRef.current, coronalMaskRef.current, 'coronal');
    render2DPane(sagittalMriRef.current, sagittalMaskRef.current, 'sagittal');
  }, [triplanarData, showVolume, labelLayers]);

  const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) } : { r: 6, g: 182, b: 212 };
  };

  const toggleLayerVisible = (id) => {
    setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, visible: !l.visible } : l));
  };

  const changeLayerColor = (id, color) => {
    setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, color } : l));
  };

  const changeLayerOpacity = (id, opacity) => {
    setLabelLayers(prev => prev.map(l => l.id === id ? { ...l, opacity: Number(opacity) } : l));
  };

  // 4. 自然语言交互驱动 Gemini API
  const handleSendMessage = () => {
    if (!inputText.trim() || isProcessing) return;
    const msgText = inputText.trim(); setInputText('');

    setChatMessages(prev => [...prev, { id: Date.now(), sender: 'user', text: msgText }]);
    setIsProcessing(true); setHarnessState('PROCESSING');

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
            layerName = `${versionId}: 左半脑实质 (Left)`; defaultColor = '#3b82f6';
          } else if (msgText.includes('右')) {
            layerName = `${versionId}: 右半脑实质 (Right)`; defaultColor = '#ef4444';
          } else if (data.action === 'SKULL_STRIP') {
            layerName = `${versionId}: 全脑实质 (Full Brain)`; defaultColor = '#06b6d4';
          }

          setLabelLayers(prev => [
            { id: versionId, name: layerName, visible: true, color: defaultColor, opacity: 0.6 },
            ...prev.filter(l => l.id !== versionId)
          ]);
        }
      })
      .catch(err => {
        setIsProcessing(false); setHarnessState('PAUSED_FOR_DOCTOR');
      });
  };

  return (
    <div className="radpilot-app">
      {/* 1. 顶部 Header (绝对锁定渲染) */}
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

      {/* 2. 主体三栏 (绝对锁定，不受中栏错误影响) */}
      <div className="app-body-triplanar">

        {/* 【左侧栏】：窗宽窗位 (WW/WL) + 3D ISO 阈值 + 图层管理器 (绝对锁定) */}
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

        {/* 【中间 4 宫格】：正交三视图 + @kitware/vtk.js vtkVolume 3D 视图 */}
        <section className="viewport-grid">
          {/* Pane 1: Axial 轴位 */}
          <div className="pane-view">
            <div className="pane-header">
              <span className="pane-badge badge-axial">AXIAL</span>
              <span>轴位 (XY)</span>
            </div>
            <div className="pane-canvas-wrapper">
              <canvas ref={axialMriRef} />
              <canvas ref={axialMaskRef} className="mask-canvas" />
            </div>
            <div className="pane-slider-bar">
              <span style={{ fontSize: 10.5, color: '#94a3b8' }}>Z:</span>
              <input type="range" min="0" max="181" value={axialIndex} onChange={(e) => setAxialIndex(Number(e.target.value))} className="pane-slider" />
              <span className="pane-slice-text">{axialIndex} / 181</span>
            </div>
          </div>

          {/* Pane 2: Coronal 冠状位 */}
          <div className="pane-view">
            <div className="pane-header">
              <span className="pane-badge badge-coronal">CORONAL</span>
              <span>冠状位 (XZ)</span>
            </div>
            <div className="pane-canvas-wrapper">
              <canvas ref={coronalMriRef} />
              <canvas ref={coronalMaskRef} className="mask-canvas" />
            </div>
            <div className="pane-slider-bar">
              <span style={{ fontSize: 10.5, color: '#94a3b8' }}>Y:</span>
              <input type="range" min="0" max="217" value={coronalIndex} onChange={(e) => setCoronalIndex(Number(e.target.value))} className="pane-slider" />
              <span className="pane-slice-text">{coronalIndex} / 217</span>
            </div>
          </div>

          {/* Pane 3: Sagittal 矢状位 */}
          <div className="pane-view">
            <div className="pane-header">
              <span className="pane-badge badge-sagittal">SAGITTAL</span>
              <span>矢状位 (YZ)</span>
            </div>
            <div className="pane-canvas-wrapper">
              <canvas ref={sagittalMriRef} />
              <canvas ref={sagittalMaskRef} className="mask-canvas" />
            </div>
            <div className="pane-slider-bar">
              <span style={{ fontSize: 10.5, color: '#94a3b8' }}>X:</span>
              <input type="range" min="0" max="181" value={sagittalIndex} onChange={(e) => setSagittalIndex(Number(e.target.value))} className="pane-slider" />
              <span className="pane-slice-text">{sagittalIndex} / 181</span>
            </div>
          </div>

              {/* Pane 4: 项目原生 DVR 3D WebGL 体渲染视场与 3D 世界坐标轴网格 */}
              <div className="pane-view">
                <div className="pane-header">
                  <span className="pane-badge badge-3d">3D DVR VOLUME</span>
                  <span>3D 体渲染 (DVR + 3D 世界坐标轴网格)</span>
                </div>
                <div className="pane-canvas-wrapper" style={{ position: 'relative', width: '100%', height: '100%' }}>
                  <ErrorBoundary>
                    <Real3DViewer
                      axialIndex={axialIndex}
                      coronalIndex={coronalIndex}
                      sagittalIndex={sagittalIndex}
                      windowWidth={windowWidth}
                      windowLevel={windowLevel}
                      volumeThreshold={volumeThreshold}
                      showVolume={showVolume}
                      maskVersion={currentVersion}
                      labelLayers={labelLayers}
                    />
                  </ErrorBoundary>
                </div>
              </div>
        </section>

        {/* 【右侧栏】：Gemini Agent 对话区 (绝对锁定渲染) */}
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
