import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// 在 WebGL / VTK.js 应用中移除 React.StrictMode，防止二次 double-mount 销毁 Canvas 与 orientationWidget
ReactDOM.createRoot(document.getElementById('root')).render(
  <App />
)
