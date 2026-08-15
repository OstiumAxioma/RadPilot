# RadPilot PACS 智能影像工作站

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.28%2B-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
![NiBabel](https://img.shields.io/badge/NiBabel-5.2%2B-E24A4A?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5.1.6-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-0.185.1-000000?style=for-the-badge&logo=threedotjs&logoColor=white)
![Gemini API](https://img.shields.io/badge/Gemini_API-Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)

<p align="center">
    <b>面向放射科医生的智能三维影像工作站与医生在环 (Human-in-the-loop) 辅助分割系统</b>
</p>

</div>

---

## 📖 项目简介

**RadPilot** 是一款结合大语言模型 (Google Gemini) 与现代 Web 三维渲染技术的放射科智能影像工作站。系统提供医疗级正交三视图 (Axial 轴位、Coronal 冠状位、Sagittal 矢状位) 与 3D 物理挤出层厚体渲染视场，支持医生通过自然语言对话下达分割、形态学微调及金标导出等全流程指令。

---

## ✨ 核心特性

- **高保真正交三视图联动**：精确适配 MNI152 标准脑空间，支持自由调节窗宽 (WW) 与窗位 (WL)，切片十字准星无缝同步。
- **3D 实体层厚体渲染与 Mask 重建**：三维空间物理层厚挤出（Slab Stack），支持三维世界空间坐标轴与正交指示线，分割结果同步在 3D 视场中立体呈现。
- **Gemini Agent 医生在环交互**：支持自然语言指令（如“提取左半脑实质”、“膨胀2像素”、“去除噪点”），自动规划并调用形态学图像算子。
- **版本控制与轨迹回溯**：支持多版本 Mask 历史回退 (Undo) 与重做 (Redo)，可一键导出 NIfTI (.nii.gz) 金标数据集。

---

## 🛠️ 技术栈与版本要求

| 模块 | 技术栈 | 推荐版本 | 说明 |
| :--- | :--- | :--- | :--- |
| **后端框架** | Python / FastAPI | Python 3.10+ / FastAPI ≥ 0.110.0 | 高性能异步医疗影像 API 服务 |
| **异步网关** | Uvicorn | ≥ 0.28.0 | ASGI 服务器 |
| **影像处理** | NiBabel / OpenCV / NumPy / Pillow | NiBabel ≥ 5.2.0, OpenCV ≥ 4.9.0 | NIfTI 医疗数据解析与 2D/3D 形态学算子 |
| **前端框架** | React / Vite | React 18.2.0 / Vite 5.1.6 | 响应式工作站 UI 与状态管理 |
| **三维渲染** | Three.js | 0.185.1 | 3D WebGL 视场、坐标轴与体渲染重建 |
| **智能模型** | Google Gemini API | Gemini Flash | 医生意图理解与算子路由分发 |

---

## 🚀 完整部署与启动指南

### 1. 环境准备

请确保本地已安装以下环境：
- **Python**: 3.10 及以上（推荐 Python 3.12）
- **Node.js**: 18.0 及以上（包含 npm 或 pnpm）
- **Git**: 用于版本管理

---

### 2. 配置 API 密钥

在项目根目录下创建 `api` 目录，并在其中创建密钥文件：

```bash
mkdir api
```

在 `api/gemini_testAPI.txt` 中填入您的 Google Gemini API Key：

```text
YOUR_GEMINI_API_KEY_HERE
```

> 💡 **提示**：`.gitignore` 已配置自动忽略 `api/` 目录，避免密钥意外提交至公共仓库。

---

### 3. 后端服务部署 (Python FastAPI)

1. 打开终端并进入项目根目录：
    ```bash
    cd d:/Project/RadPilot
    ```

2. （可选）创建并激活虚拟环境：
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux / macOS
    source .venv/bin/activate
    ```

3. 安装后端 Python 依赖：
    ```bash
    pip install -r requirements.txt
    ```

4. 启动后端 API 服务：
    ```bash
    python backend/main.py
    ```
    后端服务将运行在：`http://localhost:8000`（API 文档地址：`http://localhost:8000/docs`）

---

### 4. 前端工作站部署 (React + Vite)

1. 进入前端目录：
    ```bash
    cd d:/Project/RadPilot/frontend
    ```

2. 安装前端 npm 依赖：
    ```bash
    npm install
    ```

3. 启动前端开发服务器：
    ```bash
    npm run dev
    ```
    前端界面将运行在：`http://localhost:5173`

---

### 5. Windows 快捷一键启动

项目内置了 Windows 批处理与 PowerShell 一键启动脚本：

- **PowerShell 一键启动**：
    ```powershell
    .\start_all.ps1
    ```
- **批处理双击启动**：
    双击运行根目录下的 `start_all.bat`，即可同时唤起后端与前端服务并在浏览器中访问 `http://localhost:5173`。
- **一键终止进程**：
    双击运行 `kill_all.bat` 可快速清理后台残留的 Python 与 Node 进程。

---

## 📁 目录结构说明

```
RadPilot/
├── api/                     # API Key 存放目录 (已由 .gitignore 排除)
│   └── gemini_testAPI.txt
├── backend/                 # Python FastAPI 后端
│   ├── main.py              # 服务入口与路由端点
│   ├── image_skills.py      # NIfTI 数据加载、切片与图像算法算子
│   ├── agent_harness.py     # 医生在环 Harness 状态机与金标导出
│   ├── llm_router.py        # Gemini 提示词工程与意图解析路由
│   └── requirements.txt     # Python 依赖清单
├── frontend/                # React + Vite 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── Real3DViewer.jsx   # 3D 挤出层厚体渲染与坐标轴组件
│   │   │   └── ErrorBoundary.jsx  # 渲染错误兜底组件
│   │   ├── App.jsx          # 主工作站页面与正交四宫格视口
│   │   ├── index.css        # 全局医疗暗色主题样式
│   │   └── main.jsx         # 前端入口
│   └── package.json         # 前端依赖配置
├── image/                   # 标准医疗测试数据 (.nii)
│   └── MNI152NLin6_res-1x1x1_T1w.nii
├── export/                  # 金标标注与交互轨迹导出目录
├── .gitignore               # Git 忽略配置
├── requirements.txt         # 根目录 Python 依赖配置
├── start_all.ps1            # PowerShell 一键启动脚本
├── start_all.bat            # 批处理一键启动脚本
└── README.md                # 项目文档
```

---

## 🧪 常用自然语言测试指令

在前端右侧 Agent 对话框中输入：
- `提取全脑实质` 或 `分割全脑`
- `提取左半脑实质`
- `提取右半脑`
- `外扩2个像素` / `收缩2个像素`
- `清除孤立噪点`
- `导出金标数据`

---

## 📄 开源许可证

本项目基于 MIT 许可证开源。
