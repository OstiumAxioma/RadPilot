import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global skills_engine, llm_router, harness
    print(f"[Init] 正在载入 MRI 医疗图像: {NII_PATH}")
    skills_engine = ImageSkillsEngine(NII_PATH)
    llm_router = LLMRouter(API_KEY_PATH)
    harness = RadPilotHarness(skills_engine, llm_router)
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
        "image_path": NII_PATH,
        "slices_info": skills_engine.get_slice_count(),
        "current_version": f"v{harness.current_version_index}",
        "harness_state": harness.state
    }

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

@app.post("/api/interact")
def interact(req: InteractRequest):
    """医生自然语言交互接口"""
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt 不能为 blank")
    
    result = harness.process_doctor_input(req.prompt.strip())
    return result

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
