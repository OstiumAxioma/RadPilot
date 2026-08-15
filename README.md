# RadPilot: AI-Powered PACS Radiology Workstation

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
    <b>An Intelligent 3D Medical Imaging & Human-in-the-Loop Segmentation PACS Workstation for Radiologists</b>
</p>

</div>

---

## 📖 Overview

**RadPilot** is an intelligent radiology PACS workstation that integrates large language models (Google Gemini API) with Web-based 3D graphics rendering. It features orthogonal multi-planar reconstruction (Axial, Coronal, Sagittal) and 3D slab-extruded volume rendering with synchronized 3D segmentation mask overlays. Radiologists can interact with the workstation through natural language to perform brain extraction, morphological post-processing, and standard gold-standard dataset export.

---

## ✨ Key Features

- **Synchronized Tri-Planar Viewports**: Precise MNI152 coordinate alignment with real-time Window Width (WW) and Window Level (WL) controls, synchronized crosshair indexing across Axial, Coronal, and Sagittal planes.
- **3D Slab-Extruded Volume & Mask Reconstruction**: Three.js WebGL volume rendering with physical slab-extrusion (eliminating zero-thickness artifacts), true 3D world axes helper, and dynamic 3D mask rendering.
- **Doctor-in-the-Loop Agent Harness**: Natural language intent routing (e.g., "Extract left hemisphere brain parenchyma", "Expand 2 pixels", "Remove artifacts") powered by Gemini LLM.
- **Version Tracking & Trajectory Audit**: Multi-version mask undo/redo capabilities and one-click NIfTI (`.nii.gz`) gold-standard export.

---

## 🛠️ Tech Stack & Version Requirements

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Backend Engine** | Python / FastAPI | Python ≥ 3.10 / FastAPI ≥ 0.110.0 | High-performance asynchronous REST API |
| **ASGI Server** | Uvicorn | ≥ 0.28.0 | Production-ready ASGI web server |
| **Medical Imaging** | NiBabel / OpenCV / NumPy / Pillow | NiBabel ≥ 5.2.0, OpenCV ≥ 4.9.0 | NIfTI file parsing, 2D/3D morphological operators |
| **Frontend Framework** | React / Vite | React 18.2.0 / Vite 5.1.6 | Reactive medical workstation UI |
| **UI Design System** | Pixel-Perfect UI (Precision Crafted) | Latest | Micro-tactile 3D buttons, Inter Tight typography, precision chips |
| **3D Rendering** | @kitware/vtk.js | 36.6.2 | Native WebGL 2D orthoslices and GPU Ray-Marching 3D volume |
| **LLM Router** | Google Gemini API | Gemini Flash | Intent parsing and operator planning |

---

## 🚀 Deployment & Installation

### 1. Prerequisites

Make sure the following dependencies are installed:
- **Python**: 3.10+ (Python 3.12 recommended)
- **Node.js**: 18.0+ (with `npm` or `pnpm`)
- **Git**: For version control

---

### 2. Configure API Key

Create the `api` directory at the project root:

```bash
mkdir api
```

Create `api/gemini_testAPI.txt` and place your Google Gemini API Key inside:

```text
YOUR_GEMINI_API_KEY_HERE
```

> 🔒 **Security Note**: The `api/` directory is automatically excluded by `.gitignore` to prevent credential exposure.

---

### 3. Backend Deployment (Python FastAPI)

1. Open a terminal and navigate to the project directory:
    ```bash
    cd d:/Project/RadPilot
    ```

2. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux / macOS
    source .venv/bin/activate
    ```

3. Install backend dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Start the FastAPI backend service:
    ```bash
    python backend/main.py
    ```
    The backend server will run at: `http://localhost:8000` (API documentation: `http://localhost:8000/docs`).

---

### 4. Frontend Deployment (React + Vite)

1. Navigate to the frontend directory:
    ```bash
    cd d:/Project/RadPilot/frontend
    ```

2. Install dependencies:
    ```bash
    npm install
    ```

3. Start the Vite development server:
    ```bash
    npm run dev
    ```
    The frontend interface will run at: `http://localhost:5173`.

---

### 5. Windows Quick Start Scripts

One-click startup scripts are provided in the root directory:

- **PowerShell Launcher**:
    ```powershell
    .\start_all.ps1
    ```
- **Batch Launcher**:
    Double-click `start_all.bat` to launch both backend and frontend servers simultaneously.
- **Process Cleanup**:
    Double-click `kill_all.bat` to terminate active background Python and Node processes.

---

## 📁 Project Structure

```
RadPilot/
├── api/                     # API credentials directory (excluded by .gitignore)
│   └── gemini_testAPI.txt
├── backend/                 # Python FastAPI backend
│   ├── main.py              # Server entry point and API routes
│   ├── image_skills.py      # NIfTI loader, slicing, and morphological operators
│   ├── agent_harness.py     # Doctor-in-the-loop harness state machine
│   ├── llm_router.py        # Gemini prompt engineering and intent parsing
│   └── requirements.txt     # Backend Python dependency list
├── frontend/                # React + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Real3DViewer.jsx   # 3D volume slab rendering and axes component
│   │   │   └── ErrorBoundary.jsx  # React error boundary component
│   │   ├── App.jsx          # Main PACS workstation page and 4-pane layout
│   │   ├── index.css        # PACS dark theme stylesheet
│   │   └── main.jsx         # Frontend application entry point
│   └── package.json         # Frontend package configuration
├── image/                   # Standard medical template data (.nii)
│   └── MNI152NLin6_res-1x1x1_T1w.nii
├── export/                  # Export directory for gold-standard masks & logs
├── .gitignore               # Git ignore rules
├── requirements.txt         # Root Python requirements
├── start_all.ps1            # PowerShell launcher
├── start_all.bat            # Windows batch launcher
└── README.md                # Project documentation
```

---

## 🧪 Sample Prompts for Agent Interaction

Enter natural language commands in the right-side Agent chat box:
- `Extract full brain parenchyma`
- `Extract left hemisphere brain`
- `Extract right hemisphere`
- `Expand mask by 2 pixels` / `Shrink mask by 2 pixels`
- `Remove isolated noise artifacts`
- `Export gold standard dataset`

---

## 📄 License

This project is licensed under the MIT License.
