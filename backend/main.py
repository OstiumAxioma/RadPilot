import os
import sys
import json
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 将当前 backend 添加到 python 搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_skills import ImageSkillsEngine
from agent_core import AgentEngine, GLOBAL_TOOL_REGISTRY, VersionDAG

skills_engine: Optional[ImageSkillsEngine] = None
agent_engine: Optional[AgentEngine] = None

NII_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image", "MNI152NLin6_res-1x1x1_T1w.nii")
API_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "gemini_testAPI.txt")

current_image_path = NII_PATH
current_image_name = "MNI152NLin6_res-1x1x1_T1w.nii"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global skills_engine, agent_engine, current_image_path, current_image_name
    print(f"[Init] 正在载入默认 MRI 医疗图像: {NII_PATH}")
    skills_engine = ImageSkillsEngine(NII_PATH)
    agent_engine = AgentEngine(skills_engine.volume_data, API_KEY_PATH, spacing=skills_engine.spacing)
    current_image_path = NII_PATH
    current_image_name = "MNI152NLin6_res-1x1x1_T1w.nii"
    print("[Init] RadPilot AgentEngine (Gemini Function Calling + VersionDAG) 核心服务初始化完成！")
    yield

app = FastAPI(title="RadPilot Agent Engine API", version="2.0.0", lifespan=lifespan)

# 跨域设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    message: Optional[str] = None
    current_version: Optional[str] = None
    max_iterations: Optional[int] = 50

@app.get("/api/info")
def get_info():
    """获取图像维度、物理间距与 DAG 版本状态"""
    current_ver = agent_engine.dag.current_node_id if agent_engine else "v0"
    current_branch = agent_engine.dag.current_branch if agent_engine else "main"
    return {
        "status": "ready",
        "image_path": current_image_path,
        "image_name": current_image_name,
        "slices_info": skills_engine.get_slice_count(),
        "current_version": current_ver,
        "current_branch": current_branch,
        "voxel_spacing_mm": list(skills_engine.spacing),
        "total_dag_nodes": len(agent_engine.dag.nodes) if agent_engine else 1
    }

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """支持用户上传新的 NIfTI / DICOM 影像文件并热重载整个 PACS 与 Agent 引擎 (彻底清除旧影像与历史分割)"""
    global skills_engine, agent_engine, current_image_path, current_image_name
    try:
        import time
        filename = file.filename
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, filename)

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"[Upload] 成功接收新影像: {save_path}，正在彻底重置旧场景与 AgentEngine...")
        
        # 1. 彻底销毁旧技能引擎，加载全新单一影像
        skills_engine = ImageSkillsEngine(save_path)
        current_image_path = save_path
        current_image_name = filename
        
        # 2. 彻底重新初始化 AgentEngine 与 VersionDAG
        agent_engine = AgentEngine(skills_engine.volume_data, API_KEY_PATH, spacing=skills_engine.spacing)

        slices_info = skills_engine.get_slice_count()
        series_id = f"{filename}_{int(time.time() * 1000)}"

        return {
            "status": "success",
            "series_id": series_id,
            "image_name": filename,
            "slices_info": slices_info,
            "current_version": agent_engine.dag.current_node_id,
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
            "current_version": agent_engine.dag.current_node_id,
            "current_branch": agent_engine.dag.current_branch
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/triplanar")
def get_triplanar(axial: int = 90, coronal: int = 109, sagittal: int = 91, ww: float = 7000.0, wl: float = 3500.0):
    """三视图正交切片数据端点 (支持 MNI152 标量范围的 WW/WL 窗宽窗位)"""
    try:
        bundle = skills_engine.get_triplanar_bundle(axial, coronal, sagittal, ww=ww, wl=wl)
        bundle["current_version"] = agent_engine.dag.current_node_id
        bundle["current_branch"] = agent_engine.dag.current_branch
        return bundle
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/volume_data_vtk")
def get_volume_data_vtk():
    """供 VTK.js 前端原生 vtkImageData 构造的完整 3D MRI 标量流数据"""
    try:
        return skills_engine.get_volume_vtk_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mask_volume_vtk")
def get_mask_volume_vtk(version: Optional[str] = None):
    """供 VTK.js 前端原生 vtkImageData 构造的当前 3D Mask 标量流数据 (支持指定 DAG 节点)"""
    try:
        if version and agent_engine and version in agent_engine.dag.nodes:
            target_mask = agent_engine.dag.nodes[version].mask_data
            skills_engine.set_mask_3d(target_mask)
        elif agent_engine:
            # 默认同步当前最新激活的节点 Mask
            current_mask = agent_engine.dag.get_current_mask()
            skills_engine.set_mask_3d(current_mask)
        return skills_engine.get_mask_vtk_payload()
    except Exception as e:
        import traceback
        print(f"[Error in /api/mask_volume_vtk]: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
@app.post("/api/interact")
def interact(req: ChatRequest):
    """
    医生自然语言交互接口 (基于真实 Gemini Function Calling + Observation 闭环)
    """
    user_prompt = (req.prompt or req.message or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt 不能为空")
    
    # 真实进入 AgentEngine 决策推理与工具流
    result = agent_engine.process_user_instruction(user_prompt)
    
    # 同步最新 Mask 张量到图像技能引擎
    current_mask = agent_engine.dag.get_current_mask()
    skills_engine.set_mask_3d(current_mask)

    current_ver = result.get("current_version", agent_engine.dag.current_node_id)
    action = result.get("action", "CHAT_RESPONSE")
    reply = result.get("reply", "指令已处理。")
    
    # 动态为图层命名
    curr_node = agent_engine.dag.nodes.get(current_ver)
    layer_name = f"Mask {current_ver} ({curr_node.action_name if curr_node else action})"

    return {
        "status": "success",
        "reply": reply,
        "message": reply,
        "action": action,
        "source": result.get("source", "GEMINI_FUNCTION_CALLING"),
        "current_version": current_ver,
        "new_version": current_ver,
        "layer_name": layer_name,
        "metrics": result.get("metrics", {}),
        "executed_tools": result.get("executed_tools", []),
        "thought_steps": result.get("thought_steps", []),
        "elapsed_ms": result.get("elapsed_ms", 0),
        "current_branch": agent_engine.dag.current_branch
    }

@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """
    医生自然语言交互【实时 SSE 流式推送接口】
    实时逐帧推送每个阶段的 Thought、Action、Observation、质检告警与最新 Mask 版本！
    """
    user_prompt = (req.prompt or req.message or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt 不能为空")

    def event_generator():
        for event in agent_engine.process_user_instruction_stream(user_prompt, max_iterations=req.max_iterations):
            # 实时同步最新的 Mask 数据到 PACS 技能引擎
            if event.get("type") in ["action_done", "complete"]:
                curr_mask = agent_engine.dag.get_current_mask()
                skills_engine.set_mask_3d(curr_mask)
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/undo")
def undo():
    """撤销 Mask 版本 (DAG 回退到父节点)"""
    node = agent_engine.dag.undo()
    if node:
        skills_engine.set_mask_3d(node.mask_data)
        return {
            "status": "success",
            "current_version": node.node_id,
            "message": f"已撤销至版本节点: {node.node_id}"
        }
    return {"status": "failed", "message": "已处于初始根节点"}

@app.post("/api/redo")
def redo():
    """重做 Mask 版本 (DAG 重做至子节点)"""
    node = agent_engine.dag.redo()
    if node:
        skills_engine.set_mask_3d(node.mask_data)
        return {
            "status": "success",
            "current_version": node.node_id,
            "message": f"已重做至版本节点: {node.node_id}"
        }
    return {"status": "failed", "message": "当前分支已无更多后续版本"}

@app.get("/api/version_dag")
@app.get("/api/history")
def get_version_history():
    """获取完整的 DAG 版本图谱拓扑结构与物理度量历史"""
    return {
        "tree": agent_engine.dag.get_tree_structure() if agent_engine else [],
        "current_version": agent_engine.dag.current_node_id if agent_engine else "v0",
        "current_branch": agent_engine.dag.current_branch if agent_engine else "main"
    }

@app.get("/api/tools")
def get_available_tools():
    """获取当前注册的所有强类型医学工具 OpenAPI 规范"""
    return {
        "tools": GLOBAL_TOOL_REGISTRY.get_function_declarations()
    }

@app.post("/api/export")
def export_gold_standard():
    """导出标准 3D NIfTI 金标与 DAG 历史元数据"""
    try:
        import nibabel as nib
        export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "export")
        os.makedirs(export_dir, exist_ok=True)
        
        current_mask = agent_engine.dag.get_current_mask() if agent_engine else skills_engine.mask_data
        gold_filename = f"gold_standard_{current_image_name}"
        gold_path = os.path.join(export_dir, gold_filename)
        
        # 使用当前影像的 Affine 矩阵保存标准 NIfTI
        nii_obj = nib.Nifti1Image(current_mask.astype(np.uint8), skills_engine.nii_affine)
        nib.save(nii_obj, gold_path)
        
        return {
            "status": "success",
            "message": "金标 3D NIfTI 数据集导出成功！",
            "gold_standard_path": gold_path,
            "export_dir": export_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")

@app.get("/api/logs")
def get_recent_logs(limit: int = 10):
    """获取后端持久化的最近思维链日志文件列表与最新内容"""
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(log_dir):
            return {"logs": [], "latest_content": ""}
        
        files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        files.sort(reverse=True)
        recent_files = files[:limit]
        
        latest_content = ""
        if recent_files:
            latest_path = os.path.join(log_dir, recent_files[0])
            with open(latest_path, "r", encoding="utf-8") as lf:
                latest_content = lf.read()
                
        return {
            "log_files": recent_files,
            "latest_file": recent_files[0] if recent_files else None,
            "latest_content": latest_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
