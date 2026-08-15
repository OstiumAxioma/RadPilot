import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("3D 渲染组件捕获到局部异常:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          width: '100%', height: '100%',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          background: '#090d16', color: '#f59e0b',
          fontSize: 12, padding: 12, textAlign: 'center'
        }}>
          <span style={{ fontSize: 24, marginBottom: 6 }}>⚠️</span>
          <span>3D 视场渲染遇到异常 ({this.state.error?.message || 'WebGL 错误'})</span>
          <span style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>（主 UI、2D 视图与对话界面已受隔离保护，正常运行）</span>
          <button
            style={{ marginTop: 8, padding: '3px 8px', background: '#1e293b', border: '1px solid #334155', color: '#fff', borderRadius: 4, cursor: 'pointer', fontSize: 10 }}
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            重试加载 3D
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
