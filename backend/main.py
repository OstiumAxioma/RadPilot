import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 将当前 backend 添加到 python 搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_skills import ImageSkillsEngine
from llm_router import LLMRouter
from agent_harness import RadPilotHarness

skills_engine = None
llm_router = None
harness = None

NII_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image", "MNI152NLin6_res-1x1x1_T1w.nii")
API_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "gemini_testAPI.txt")

current_image_path = NII_PATH
current_image_name = "MNI152NLin6_res-1x1x1_T1w.nii"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global skills_engine, llm_router, harness, current_image_path, current_image_name
    print(f"[Init] 正在载入默认 MRI 医疗图像: {NII_PATH}")
    skills_engine = ImageSkillsEngine(NII_PATH)
    llm_router = LLMRouter(API_KEY_PATH)
    harness = RadPilotHarness(skills_engine, llm_router)
    current_image_path = NII_PATH
    current_image_name = "MNI152NLin6_res-1x1x1_T1w.nii"
    print("[Init] RadPilot Agent Harness 后端服务初始化完成！")
    yield

app = FastAPI(title="RadPilot Agent Harness API", version="1.0.0", lifespan=lifespan)

# 跨域设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InteractRequest(BaseModel):
    prompt: str

class ToolCallRequest(BaseModel):
    action: str
    pixels: Optional[int] = 2

@app.get("/api/info")
def get_info():
    """获取图像维度与状态信息"""
    return {
        "status": "ready",
        "image_path": current_image_path,
        "image_name": current_image_name,
        "slices_info": skills_engine.get_slice_count(),
        "current_version": f"v{harness.current_version_index}",
        "harness_state": harness.state
    }

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """支持用户上传新的 NIfTI / DICOM 影像文件并热重载整个 PACS 引擎 (彻底清除旧影像与历史分割)"""
    global skills_engine, harness, current_image_path, current_image_name
    try:
        import time
        filename = file.filename
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, filename)

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"[Upload] 成功接收新影像: {save_path}，正在彻底重置旧场景与引擎...")
        
        # 1. 彻底销毁旧技能引擎，加载全新单一影像
        skills_engine = ImageSkillsEngine(save_path)
        current_image_path = save_path
        current_image_name = filename
        
        # 2. 彻底重置 Harness 状态机（清空所有旧版本树与轨迹，回归 v0 纯净态）
        harness = RadPilotHarness(skills_engine, llm_router)
        harness.version_tree = {}
        harness.current_version_index = 0
        harness.trajectory_events = []
        harness.state = "PAUSED_FOR_DOCTOR"

        slices_info = skills_engine.get_slice_count()
        series_id = f"{filename}_{int(time.time() * 1000)}"

        return {
            "status": "success",
            "series_id": series_id,
            "image_name": filename,
            "slices_info": slices_info,
            "current_version": "v0",
            "message": f"已彻底重置场景并载入唯一新序列: {filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"影像文件载入失败: {str(e)}")

@app.get("/api/slice")
def get_slice(index: int = Query(90, ge=0), axis: str = Query("axial")):
    """获取指定切片的底图灰度图 base64 和当前 Mask 切片 base64"""
    try:
        slice_b64 = skills_engine.get_slice_base64(index, axis)
        mask_b64 = skills_engine.get_mask_slice_base64(index, axis)
        return {
            "slice_index": index,
            "axis": axis,
            "image_base64": slice_b64,
            "mask_base64": mask_b64,
            "current_version": f"v{harness.current_version_index}",
            "harness_state": harness.state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/triplanar")
def get_triplanar(axial: int = 90, coronal: int = 109, sagittal: int = 91, ww: float = 7000.0, wl: float = 3500.0):
    """三视图正交切片数据端点 (支持 MNI152 标量范围的 WW/WL 窗宽窗位)"""
    try:
        bundle = skills_engine.get_triplanar_bundle(axial, coronal, sagittal, ww=ww, wl=wl)
        bundle["current_version"] = f"v{harness.current_version_index}"
        bundle["harness_state"] = harness.state
        return bundle
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/volume_data_raw")
def get_volume_data_raw():
    """供 3D 体渲染管线加载的标量体数据端点"""
    try:
        return skills_engine.get_volume_raw_base64()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mask_volume_raw")
def get_mask_volume_raw():
    """供 3D 体渲染管线加载的 Mask 体数据端点 (与体数据坐标系完全一致)"""
    try:
        return skills_engine.get_mask_volume_raw_base64()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/volume_data_vtk")
def get_volume_data_vtk():
    """供 VTK.js 统一初始化的 3D ImageData 载荷端点"""
    try:
        return skills_engine.get_volume_vtk_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mask_volume_vtk")
def get_mask_volume_vtk():
    """供 VTK.js 统一初始化的 3D Mask ImageData 载荷端点"""
    try:
        return skills_engine.get_mask_vtk_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/volume_atlas")
def get_volume_atlas():
    """3D GPU Raymarching 体数据 2D Atlas 端点"""
    try:
        return {
            "status": "success",
            "atlas_base64": skills_engine.get_volume_atlas_base64(),
            "cols": 10,
            "rows": 10,
            "slices": 96
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    message: Optional[str] = None
    current_version: Optional[str] = None

@app.post("/api/chat")
@app.post("/api/interact")
def interact(req: ChatRequest):
    """医生自然语言交互接口 (同时支持 /api/chat 与 /api/interact)"""
    user_prompt = (req.prompt or req.message or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt 不能为 blank")
    
    result = harness.process_doctor_input(user_prompt)
    
    # 格式化兼容返回值给前端
    current_ver = result.get("current_version", f"v{harness.current_version_index}")
    action = result.get("action", "INSPECT")
    msg = result.get("message", "指令已成功解析并执行。")
    
    layer_map = {
        "SKULL_STRIP": "脑实质分割图层",
        "EXPAND": "Mask 外扩图层",
        "SHRINK": "Mask 收缩图层",
        "REMOVE_ARTIFACTS": "去伪影图层",
        "INVERT": "反相 Mask 图层",
        "RESET": "重置 Mask"
    }
    layer_name = layer_map.get(action, f"分割图层 ({current_ver})")

    return {
        "status": result.get("status", "success"),
        "reply": msg,
        "message": msg,
        "action": action,
        "action_type": action,
        "source": result.get("source", "GEMINI_ROUTER"),
        "current_version": current_ver,
        "new_version": current_ver if result.get("status") == "success" and action not in ["UNKNOWN", "INSPECT"] else None,
        "layer_name": layer_name,
        "elapsed_ms": result.get("elapsed_ms", 0),
        "state": harness.state
    }

@app.post("/api/tool")
def execute_tool(req: ToolCallRequest):
    """快捷工具调用接口"""
    action_map = {
        "skull_strip": "SKULL_STRIP",
        "expand": "EXPAND",
        "shrink": "SHRINK",
        "remove_artifacts": "REMOVE_ARTIFACTS",
        "invert": "INVERT",
        "reset": "RESET"
    }
    
    act = action_map.get(req.action.lower(), req.action.upper())
    prompt_desc = f"快捷工具操作: {act}"
    if act in ["EXPAND", "SHRINK"]:
        prompt_desc += f" {req.pixels}像素"
        
    result = harness.process_doctor_input(prompt_desc)
    return result

@app.post("/api/undo")
def undo():
    """撤销 Mask 版本"""
    return harness.undo()

@app.post("/api/redo")
def redo():
    """重做 Mask 版本"""
    return harness.redo()

@app.get("/api/history")
def get_history():
    """获取 Mask 版本历史树与轨迹信息"""
    return {
        "version_tree": harness.get_version_tree(),
        "current_index": harness.current_version_index,
        "trajectory_count": len(harness.trajectory_events),
        "harness_state": harness.state
    }

@app.post("/api/export")
def export_gold_standard():
    """导出标准 3D NIfTI 金标与 Trajectory"""
    try:
        export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "export")
        gold_path = harness.export_gold_standard(export_dir)
        return {
            "status": "success",
            "message": "金标数据集与 Trajectory 日志导出成功！",
            "gold_standard_path": gold_path,
            "export_dir": export_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
