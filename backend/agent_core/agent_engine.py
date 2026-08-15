import os
import json
import time
import requests
import numpy as np
from scipy import ndimage
from typing import Dict, List, Any, Optional, Generator

from .tools.base_tool import ImageContext, ToolResult
from .tools.registry import GLOBAL_TOOL_REGISTRY, ToolRegistry
from .version_dag import VersionDAG, VersionNode
from .multimodal.slice_encoder import MultiModalSliceEncoder
from .logger import GLOBAL_THOUGHT_LOGGER, AgentThoughtLogger

class AgentEngine:
    """
    RadPilot 专业医学影像 ReAct 自主推理引擎
    支持流式推送 (SSE) 与【执行-质检验收-精修】强闭环状态机
    """
    def __init__(
        self,
        image_data: np.ndarray,
        api_key_path: str = "api/gemini_testAPI.txt",
        spacing: tuple = (1.0, 1.0, 1.0),
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 50
    ):
        self.image_data = image_data
        self.spacing = spacing
        self.tool_registry = tool_registry or GLOBAL_TOOL_REGISTRY
        self.dag = VersionDAG(image_data.shape)
        self.max_iterations = max_iterations
        
        self.api_key = self._load_api_key(api_key_path)
        self.model_name = "gemini-flash-latest"
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def _load_api_key(self, path: str) -> str:
        """从文件读取 API Key"""
        if not os.path.exists(path):
            print(f"[AgentEngine Warning] API Key 文件不存在: {path}")
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("AQ."):
                        return line
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                if lines:
                    return lines[-1]
        except Exception as e:
            print(f"[AgentEngine Warning] 读取 API Key 异常: {e}")
        return ""

    def get_current_context(self) -> ImageContext:
        """构建当前最新的物理图像上下文"""
        current_mask = self.dag.get_current_mask()
        return ImageContext(
            image_data=self.image_data,
            current_mask=current_mask,
            spacing=self.spacing
        )

    def process_user_instruction_stream(self, user_prompt: str, max_iterations: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        核心 ReAct 自主精修与质检验收流式生成器 (Streaming ReAct Loop)
        向前端逐帧实时推送每个阶段的 Thought、Action、Observation 与质检结论！
        """
        start_time = time.time()
        context = self.get_current_context()
        env_summary = context.get_summary()
        limit_steps = max_iterations if (max_iterations is not None and max_iterations > 0) else self.max_iterations

        # 处理快捷指令 (Undo / Redo)
        prompt_lower = user_prompt.strip().lower()
        if prompt_lower in ["撤销", "undo", "上一步", "后退"]:
            prev_node = self.dag.undo()
            elapsed = int((time.time() - start_time) * 1000)
            if prev_node:
                yield {
                    "type": "complete",
                    "reply": f"已成功撤销至历史版本 [{prev_node.node_id}] ({prev_node.action_name})。当前标定体积: {prev_node.metrics.get('current_volume_cm3', 0)} cm³。",
                    "action": "UNDO",
                    "source": "VERSION_DAG",
                    "current_version": prev_node.node_id,
                    "new_version": prev_node.node_id,
                    "metrics": prev_node.metrics,
                    "elapsed_ms": elapsed,
                    "thought_steps": []
                }
            else:
                yield {
                    "type": "complete",
                    "reply": "已处于版本树的初始根节点 (v0)，无法继续撤销。",
                    "action": "UNDO_FAILED",
                    "source": "VERSION_DAG",
                    "current_version": self.dag.current_node_id,
                    "elapsed_ms": elapsed,
                    "thought_steps": []
                }
            return

        if prompt_lower in ["重做", "redo", "下一步", "前进"]:
            next_node = self.dag.redo()
            elapsed = int((time.time() - start_time) * 1000)
            if next_node:
                yield {
                    "type": "complete",
                    "reply": f"已成功重做至版本分支 [{next_node.node_id}] ({next_node.action_name})。当前标定体积: {next_node.metrics.get('current_volume_cm3', 0)} cm³。",
                    "action": "REDO",
                    "source": "VERSION_DAG",
                    "current_version": next_node.node_id,
                    "new_version": next_node.node_id,
                    "metrics": next_node.metrics,
                    "elapsed_ms": elapsed,
                    "thought_steps": []
                }
            else:
                yield {
                    "type": "complete",
                    "reply": "当前分支已是最新版本节点，没有可重做的后续历史。",
                    "action": "REDO_FAILED",
                    "source": "VERSION_DAG",
                    "current_version": self.dag.current_node_id,
                    "elapsed_ms": elapsed,
                    "thought_steps": []
                }
            return

        if not self.api_key:
            yield {
                "type": "complete",
                "reply": "未检测到有效的 Gemini API 密钥，请检查 api/gemini_testAPI.txt 配置。",
                "action": "AUTH_ERROR",
                "source": "AGENT_ENGINE",
                "current_version": self.dag.current_node_id,
                "elapsed_ms": int((time.time() - start_time) * 1000),
                "thought_steps": []
            }
            return

        # 记录本次自然语言交互启动日志
        GLOBAL_THOUGHT_LOGGER.log_interaction_start(user_prompt, env_summary)

        system_instruction = (
            "你是一个具备原生多模态视觉空间推理与【全自由度视觉诊断 - 智能工具匹配 - 闭环精修】的资深放射学智能体 RadPilot。\n\n"
            "【三维正交多视角 (Tri-Planar) 空间协同输入】:\n"
            "系统在每轮迭代为你提供了当前三维医学体数据与最新 Mask 叠加的正交三视角中心断层画廊（青色高亮区域为当前已标定掩码）：\n"
            "1. 【图1: 轴位横断面 Axial (Z 轴)】: 从颅底/小脑(Z=18%)、第四脑室(Z=30%)到半卵圆中心(Z=68%)、顶叶(Z=82%)。\n"
            "   -> 💡 临床优势: 审视与精修【左/右对称性、双侧半球外缘、小脑横向叶段、侧脑室张开度】的最佳视角！\n"
            "2. 【图2: 冠状位额状面 Coronal (Y 轴)】: 前额叶(Y=30%)、中线脑干(Y=50%)、后颅窝小脑/枕叶(Y=70%)。\n"
            "   -> 💡 临床优势: 审视与精修【颅顶至颅底高度、小脑幕与枕叶上下分界、鞍区/脑干腹侧】的最佳视角！\n"
            "3. 【图3: 矢状位正中面 Sagittal (X 轴)】: 右半球(X=30%)、正中矢状面(X=50%)、左半球(X=70%)。\n"
            "   -> 💡 临床优势: 审视与精修【前额至后枕前后径、脑干背侧、第四脑室底与小脑蚓部】的最佳视角！\n\n"
            f"【物理空间元数据】:\n"
            f"- 空间维度 (Shape: X, Y, Z): {env_summary['image_dimensions']}\n"
            f"- 物理体素间距 (Spacing mm): {env_summary['voxel_spacing_mm']}\n"
            f"- 单个体素物理体积: {env_summary['voxel_volume_mm3']} mm³\n"
            f"- 当前已有 Mask 标定体积: {env_summary['mask_volume_cm3']} cm³\n"
            f"- 影像信号强度范围: [{env_summary['image_min_intensity']}, {env_summary['image_max_intensity']}], 均值: {env_summary['image_mean_intensity']}\n\n"
            "【放射学专家三视角协同诊断与工具箱决策法则】:\n"
            "严禁仅在单一视角打转！请像资深放射科专家一样，在【轴位 Axial】、【冠状位 Coronal】与【矢状位 Sagittal】三视角间积极联动与切换：\n"
            "0. 【多视角切片跳转与局部高清变焦】: "
            "   -> 若需要查看轴位特定横断面（如小脑最宽层 Z=35）、冠状面（如后颅窝 Y=55）或矢状面，调用 `inspect_ortho_slice(plane='axial'|'coronal'|'sagittal', slice_index=...)` 或 `browse_slice_gallery`，系统会立即将该视角高清图像注入你的视觉中！\n"
            "1. 【主动连通性深度拓扑诊断】: "
            "   -> 💡 调用 `analyze_connectivity` 深度扫描当前掩码，直接获取所有独立连通域的详细档案（体积、质心三维坐标 [X,Y,Z]、切片跨度），精准定位哪块是主器官、哪块是粘连碎屑！\n"
            "2. 【三视角连续曲线多边形绘制 (VTK 光栅化)】: "
            "   -> 在轴位调用 `draw_polygon_contour(plane='axial', slice_index=Z, points=[[X,Y],...])` 勾画左右横断面；"
            "   -> 在冠状位调用 `draw_polygon_contour(plane='coronal', slice_index=Y, points=[[X,Z],...])` 勾画上下额状面；"
            "   -> 在矢状位调用 `draw_polygon_contour(plane='sagittal', slice_index=X, points=[[Y,Z],...])` 勾画前后矢状面；\n"
            "2. 【分水岭解剖图割 (Watershed / GrowCut)】: "
            "   -> 💡 仿照 3D Slicer GrowCut 与 Photoshop 快速选择: 调用 `watershed_segmentation` 传入前景点 (目标内部) 与背景点 (周围脑干/水腔/颅骨)，沿梯度地形图自动生成精准贴合自然解剖纹理的掩码！\n"
            "3. 【PS 智能边缘吸附画笔 (Smart Intensity Brush)】: "
            "   -> 💡 仿照 Photoshop 智能磁性画笔: 调用 `smart_intensity_brush(plane=..., slice_index=..., point_2d=[c1,c2], radius_mm=8.0)` 涂抹时自动仅吸收同类组织信号，并在遇解剖边界/脑脊液/颅骨时自动智能吸附贴边阻断！\n"
            "4. 【三视角 VTK 曲线剪刀与解剖剥离】: "
            "   -> 沿轴位、冠状位或矢状位调用 `contour_scissors_cut` 剪除特定断面的粘连（如在冠状位剪除小脑幕上方枕叶，在轴位剪除脑桥前缘）；\n"
            "5. 【通用空间多模态引导与区域生长】: "
            "   -> 调用 `spatial_prompt_guided_segmentation` 或 `region_growth` 传入器官三维中心点/包载盒进行自适应解剖提取；\n"
            "6. 【多断层关键帧轮廓插值】: "
            "   -> 在多个轴位/冠状位断层绘制关键轮廓后调用 `fill_between_slices` 沿 3D 测地线曲面平滑重构；\n"
            "7. 【组织信号强度交集/并集卡取】: "
            "   -> 调用 `segment_by_intensity_range` 或 `threshold_range(mode='intersect')` 剥离颅外高亮脂肪与低信号骨板；\n"
            "8. 【微调精修首选: 局部橡皮擦 (Erase Brush)】: "
            "   -> 💡 当掩码只差一点点微小局灶粘连、边缘突刺、中线小毛刺时，严禁使用大剪刀粗暴全切！直接在当前切片调用 `erase_brush_3d(plane='axial'|'coronal'|'sagittal', slice_index=..., point_2d=[c1,c2], radius_mm=3.0)` 局部擦除！\n"
            "9. 【微调精修首选: 局部画笔 (Paint Brush)】: "
            "   -> 💡 当掩码仅有边缘局部漏包、微小孔洞或叶尖遗漏时，严禁重拉全局阈值！直接在当前切片调用 `paint_brush_3d(plane=..., slice_index=..., point_2d=[c1,c2], radius_mm=3.0)` 局部补点！\n"
            "10. 【离散卫星碎屑 / 内部闭合微孔】: "
            "   -> 调用 `island_and_smooth` 过滤杂质并充填内部空腔；\n"
            "11. 【切片阶梯毛刺 / 边缘锯齿】: "
            "   -> 调用 `morphological_smooth` 进行亚毫米级平滑。\n\n"
            "【允许坦承工具链缺失与提出脚手架改进需求】:\n"
            "如果你审视影像后发现当前脚手架提供的通用工具无法满足特定的复杂解剖要求（例如缺少某种 3D 测地线水平集演化算子、缺少斜断面剪刀、或当前结构无法仅靠现有工具精准分割），"
            "你完全可以直接输出 `【💡 工具链需求与优化建议】` 详细说明你需要的工具能力与算法缺失原因，无需强行调用不适用的工具！系统会完整记录你的建议以指导底层工具链的升级！\n\n"
            "【ReAct 思维链输出规范】:\n"
            "在每轮调用工具前，必须输出一段详尽的【Thought】，格式包含：\n"
            "  - 视觉观察: 在哪个切片视角（轴位 Z、冠状位 Y、矢状位 X）观察到了什么解剖特征或缺陷；\n"
            "  - 决策理由: 为什么选择该工具及在该视角下操作；\n"
            "  - 预期目标: 本步骤期望达到的解剖改善效果。\n"
            "当且仅当在轴位、冠状位与矢状位三视角全面交叉核对无瑕、无粘连、无漏割、无孤岛且生理体积指标完全正常时，方可输出终审临床评估报告！"
        )

        tools_declarations = self.tool_registry.get_function_declarations()

        # 初始动态截取切片图像帧
        multiview_image_parts = MultiModalSliceEncoder.encode_multiview_slices(
            self.image_data,
            current_mask=context.current_mask
        )

        conversation_contents = [
            {
                "role": "user",
                "parts": multiview_image_parts + [{"text": user_prompt}]
            }
        ]

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }

        thought_steps = []
        final_clinical_reply = ""
        last_action_name = "INSPECT"
        last_node_id = self.dag.current_node_id
        current_warnings = ["初始状态，尚未生成解剖掩码"]

        # -------------------------------------------------------------
        # ReAct 多轮循环状态机 (支持由粗到细 1~8 步深度精雕细琢)
        # -------------------------------------------------------------
        for iteration in range(1, limit_steps + 1):
            step_start = time.time()
            
            # 推送本轮步骤开始事件
            yield {
                "type": "step_start",
                "step_index": iteration,
                "iteration": iteration,
                "status": "THINKING",
                "message": f"正在审视三维正交画廊并进行第 {iteration} 轮解剖推理与决策..."
            }

            payload = {
                "contents": conversation_contents,
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                },
                "tools": [
                    {
                        "functionDeclarations": tools_declarations
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1024
                }
            }

            try:
                # 发送多模态视觉推理请求 (支持最大 2 次指数退避重试，单次超时 90s)
                res_json = None
                max_retries = 2
                for attempt in range(1, max_retries + 1):
                    try:
                        res = requests.post(self.endpoint, headers=headers, json=payload, timeout=90)
                        if res.status_code == 200:
                            res_json = res.json()
                            break
                        else:
                            print(f"[Gemini API Attempt {attempt} Failed] HTTP {res.status_code}: {res.text}")
                            if attempt < max_retries:
                                time.sleep(2)
                    except requests.exceptions.Timeout:
                        print(f"[Gemini API Attempt {attempt} Timeout] 请求超时 (90s)，正在重试...")
                        if attempt < max_retries:
                            time.sleep(2)
                    except Exception as net_err:
                        print(f"[Gemini API Attempt {attempt} Error] 网络异常: {net_err}")
                        if attempt < max_retries:
                            time.sleep(2)

                if not res_json:
                    final_clinical_reply = "连接 Gemini 多模态推理服务器超时或网络中断，请检查网络代理与 API 密钥后重试。"
                    break

                candidates = res_json.get("candidates", [])
                if not candidates:
                    final_clinical_reply = "模型未返回有效候选内容。"
                    break

                candidate_content = candidates[0].get("content", {})
                parts = candidate_content.get("parts", [])

                tool_calls = []
                text_chunks = []
                for p in parts:
                    if "functionCall" in p:
                        tool_calls.append(p["functionCall"])
                    elif "text" in p:
                        text_chunks.append(p["text"])

                current_thought = "\n".join(text_chunks).strip()

                # 推送 Thought 事件
                if current_thought:
                    yield {
                        "type": "thought",
                        "step_index": iteration,
                        "thought": current_thought
                    }

                # 【强质检验收与工具链缺失自白处理】: 若无 Tool 调用
                if not tool_calls:
                    curr_mask = self.dag.get_current_mask()
                    curr_vol = float(np.count_nonzero(curr_mask) * context.voxel_volume_mm3 / 1000.0)
                    
                    # 检查模型是否明确提出了工具链缺失诉求或算法改进建议
                    is_tool_feedback = any(k in current_thought for k in [
                        "工具链", "缺少工具", "建议增加", "无法通过现有工具", "未提供", "不足以", "无法实现", "无法完成"
                    ])

                    if is_tool_feedback:
                        print(f"[Agent Toolchain Feedback]: 模型坦承现有工具链不足并提出改进需求: {current_thought[:100]}...")
                        final_clinical_reply = current_thought
                        break

                    # 若存在质检告警，或者第一轮未做任何操作，拦截提前交卷！
                    if current_warnings or iteration == 1 or curr_vol <= 0.0:
                        warn_desc = "; ".join(current_warnings) if current_warnings else "掩码体积为 0 或尚未完成完整精修"
                        print(f"[ReAct Quality Gate Interlock] 拦截提前退出: {warn_desc}")
                        
                        overlay_slices = MultiModalSliceEncoder.encode_multiview_slices(
                            self.image_data,
                            current_mask=curr_mask
                        )
                        conversation_contents.append(candidate_content)
                        conversation_contents.append({
                            "role": "user",
                            "parts": overlay_slices + [{
                                "text": (
                                    f"【系统强质检验收警报拦截 (Iteration {iteration})】:\n"
                                    f"检测到你仅输出了观察分析文本，但尚未下发任何工具调用 (functionCall)！\n"
                                    f"当前质检指标尚未通过验收: 【{warn_desc}】。\n"
                                    f"请根据你的观察下发通用工具（如 `contour_scissors_cut` 剪除粘连，或 `draw_polygon_contour` 补全漏割）。\n"
                                    f"若你认为当前脚手架确实缺少所需算法算子，请显式输出【💡 工具链需求与优化建议】说明原因以终止流程！"
                                )
                            }]
                        })
                        continue  # 强制驱动进入下一轮迭代！

                    # 质检合格且有真实结果
                    final_clinical_reply = current_thought or "已完成解剖分割并通过质检验收。"
                    break

                # 依次执行下发的工具
                for call in tool_calls:
                    tool_name = call.get("name")
                    tool_args = call.get("args", {})
                    last_action_name = tool_name.upper()

                    # 推送 Action 启动事件
                    yield {
                        "type": "action_start",
                        "step_index": iteration,
                        "action_name": tool_name,
                        "action_params": tool_args
                    }

                    # 执行工具
                    current_ctx = self.get_current_context()
                    tool_res = self.tool_registry.execute_tool(tool_name, current_ctx, **tool_args)
                    
                    if tool_res.success:
                        new_node = self.dag.commit(
                            action_name=f"{tool_name.upper()}_S{iteration}",
                            prompt=user_prompt,
                            new_mask=tool_res.new_mask,
                            metrics=tool_res.observation_metrics,
                            tool_name=tool_name,
                            tool_args=tool_args
                        )
                        last_node_id = new_node.node_id

                    step_elapsed = int((time.time() - step_start) * 1000)

                    # 质控自检
                    curr_mask = self.dag.get_current_mask()
                    labeled_islands, island_count = ndimage.label(curr_mask > 0)
                    current_vol_cm3 = tool_res.observation_metrics.get('current_volume_cm3', 0.0)
                    dim_x, dim_y, dim_z = context.shape
                    mid_x = dim_x // 2

                    obs_summary = (
                        f"已执行 {tool_res.message}。"
                        f"变化体积: {tool_res.observation_metrics.get('volume_change_mm3', 0)} mm³，"
                        f"当前标定总体积: {current_vol_cm3} cm³。"
                        f"【质检指标】独立连通分支数: {island_count}。"
                    )
                    
                    step_info = {
                        "step_index": len(thought_steps) + 1,
                        "iteration": iteration,
                        "thought": current_thought or f"执行阶段 {iteration}: 调用 {tool_name} 进行解剖计算",
                        "action_name": tool_name,
                        "action_params": tool_args,
                        "observation": obs_summary,
                        "metrics": tool_res.observation_metrics,
                        "mask_version": last_node_id,
                        "elapsed_ms": step_elapsed
                    }
                    thought_steps.append(step_info)

                    # 推送 Action 完成与 Observation 指标事件 (前端此时可立即呈现该步骤与更新视图！)
                    yield {
                        "type": "action_done",
                        "step_index": iteration,
                        "step_data": step_info,
                        "current_version": last_node_id
                    }

                    # 过分割与欠分割双向生理容积与解剖越界深度质检
                    leakage_warnings = []
                    if any(k in user_prompt for k in ["左脑", "左半球", "left"]):
                        overflow_right = int(np.count_nonzero(curr_mask[:mid_x, :, :] > 0))
                        if overflow_right > 50:
                            leakage_warnings.append(f"【中线过分割越界】: 检测到掩码越过正中矢状面侵入右脑 {overflow_right} 个体素，必须调用 `scissors_cut(plane='sagittal', cut_index={mid_x}, remove_side='less_than')` 切除！")
                        if current_vol_cm3 > 750.0:
                            leakage_warnings.append(f"【生理容积过分割】: 当前体积 ({current_vol_cm3} cm³) 显著超出成年人单侧半球正常生理范围 (500~700 cm³)，疑似包入了颅底软组织或对侧脑，必须使用 `scissors_cut` 或 `erase_brush_3d` 剔除多余部分！")
                        elif current_vol_cm3 < 450.0 and current_vol_cm3 > 0:
                            leakage_warnings.append(f"【生理容积欠分割】: 当前体积 ({current_vol_cm3} cm³) 低于成年人单侧半球正常生理范围 (500~700 cm³)，存在大量脑实质遗漏，必须补全！")

                    elif any(k in user_prompt for k in ["右脑", "右半球", "right"]):
                        overflow_left = int(np.count_nonzero(curr_mask[mid_x:, :, :] > 0))
                        if overflow_left > 50:
                            leakage_warnings.append(f"【中线过分割越界】: 检测到掩码越过正中矢状面侵入左脑 {overflow_left} 个体素，必须调用 `scissors_cut(plane='sagittal', cut_index={mid_x}, remove_side='greater_than')` 切除！")
                        if current_vol_cm3 > 750.0:
                            leakage_warnings.append(f"【生理容积过分割】: 当前体积 ({current_vol_cm3} cm³) 显著超出成年人单侧半球正常生理范围 (500~700 cm³)，必须修剪！")
                        elif current_vol_cm3 < 450.0 and current_vol_cm3 > 0:
                            leakage_warnings.append(f"【生理容积欠分割】: 当前体积 ({current_vol_cm3} cm³) 低于单侧半球正常生理范围 (500~700 cm³)，必须补全！")

                    elif any(k in user_prompt for k in ["小脑", "cerebellum"]):
                        overflow_z = int(np.count_nonzero(curr_mask[:, :, int(dim_z * 0.42):] > 0))
                        if overflow_z > 50:
                            leakage_warnings.append(f"【向上过分割】: 小脑掩码向上溢出侵入枕叶 ({overflow_z} 个体素)，必须调用 `scissors_cut(plane='axial', cut_index={int(dim_z*0.40)}, remove_side='greater_than')` 切除！")
                        if current_vol_cm3 > 200.0:
                            leakage_warnings.append(f"【小脑生理容积过分割】: 当前体积 ({current_vol_cm3} cm³) 显著超出成人小脑生理范围 (130~170 cm³)，存在严重过度包绕！")
                        elif current_vol_cm3 < 120.0 and current_vol_cm3 > 0:
                            leakage_warnings.append(f"【小脑生理容积欠分割】: 当前体积 ({current_vol_cm3} cm³) 显著低于正常成人小脑范围 (130~170 cm³)，明显遗漏了小脑半球后叶或小脑蚓部，禁止提前交卷，必须补全！")

                    if island_count > 1:
                        leakage_warnings.append(f"【孤立碎屑伪影】: 存在 {island_count} 个独立不连通的离散孤岛，建议调用 `island_and_smooth` 过滤杂质。")

                    overlay_slices = MultiModalSliceEncoder.encode_multiview_slices(
                        self.image_data,
                        current_mask=curr_mask
                    )

                    warning_text = "\n".join([f"- ⚠️ {w}" for w in leakage_warnings]) if leakage_warnings else "- ✅ 各项解剖边界与生理容积指标质检正常，未检测到明显过分割或中线泄漏。"
                    current_warnings = leakage_warnings.copy()

                    # 推送质检验收门控触发事件
                    yield {
                        "type": "verification_gate",
                        "step_index": iteration,
                        "warnings": leakage_warnings,
                        "island_count": island_count,
                        "volume_cm3": current_vol_cm3
                    }

                    inspection_prompt = (
                        f"【系统放射学质检验收与多维自检反馈 (Iteration {iteration})】:\n"
                        f"- 刚刚执行的工具: [{tool_name}]\n"
                        f"- 当前标定物理体积: {current_vol_cm3} cm³\n"
                        f"- 独立连通域孤岛数: {island_count} 个\n"
                        f"{warning_text}\n\n"
                        "请全面审视附带的图1轴位(Z)、图2冠状位(Y)与图3矢状位(X)最新正交切片画廊进行放射学临床视觉验收审查：\n"
                        "1. 【三视角解剖边界贴合度】: 观察各个断层切片上的青色掩码是否严密包覆目标器官边缘。注意在【轴位 Z】核查左右对称性与外缘，在【冠状位 Y】核查颅底与顶部高度，在【矢状位 X】核查前后分界与脑干粘连；\n"
                        "2. 【跨视角自主工具决策】: 若发现粘连可在对应视角调用 `contour_scissors_cut`/`erase_brush_3d`；若发现断层漏割可切换至该切片调用 `draw_polygon_contour(plane='axial'|'coronal'|'sagittal', mode='add')` 进行补全；\n"
                        "3. 【终审合格判据】: 只有在轴位、冠状位与矢状位三视角全面交叉核对精准无瑕、无粘连、无漏割、无孤岛且生理容积指标完全符合人体解剖时，方可输出最终临床评估定量总结报告！"
                    )

                    # 记录本轮 ReAct 完整步骤日志 (Thought, Action, Observation, Verification)
                    GLOBAL_THOUGHT_LOGGER.log_step(
                        iteration=iteration,
                        thought=current_thought,
                        tool_calls=[{"name": tool_name, "args": tool_args}],
                        tool_results=[{"message": tool_res.message, "metrics": tool_res.observation_metrics}],
                        verification_feedback=inspection_prompt,
                        elapsed_ms=step_elapsed
                    )

                    conversation_contents.append(candidate_content)
                    conversation_contents.append({
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {
                                        "status": "success" if tool_res.success else "error",
                                        "observation_metrics": tool_res.observation_metrics,
                                        "island_count": island_count,
                                        "message": tool_res.message
                                    }
                                }
                            }
                        ] + ([tool_res.attached_image_part] if getattr(tool_res, 'attached_image_part', None) else overlay_slices) + [{"text": inspection_prompt}]
                    })

            except Exception as e:
                import traceback
                print(f"[ReAct Loop Exception]: {traceback.format_exc()}")
                final_clinical_reply = f"ReAct 推理循环异常: {str(e)}"
                break

        if not final_clinical_reply:
            final_clinical_reply = f"已通过 ReAct 验收闭环完成精细化勾画，共执行 {len(thought_steps)} 步迭代。"

        # 【0 体积自动回滚保护】: 若循环结束时当前掩码恰好为 0 体积 (如因 reset 或过度修剪)，自动回滚至历史最佳非零节点
        curr_final_mask = self.dag.get_current_mask()
        curr_final_vol = float(np.count_nonzero(curr_final_mask) * context.voxel_volume_mm3 / 1000.0)
        if curr_final_vol <= 0.0 and len(self.dag.nodes) > 1:
            best_hist_node = None
            max_v = 0.0
            for n_id, n in self.dag.nodes.items():
                if n.mask is not None:
                    v = float(np.count_nonzero(n.mask) * context.voxel_volume_mm3 / 1000.0)
                    if v > max_v:
                        max_v = v
                        best_hist_node = n
            if best_hist_node and max_v > 0.0:
                print(f"[Rollback Guard] 最终掩码为 0，已自动为您恢复至最佳历史掩码节点 {best_hist_node.id} (体积: {max_v} cm³)")
                self.dag.checkout(best_hist_node.id)
                last_node_id = best_hist_node.id

        total_elapsed = int((time.time() - start_time) * 1000)
        current_node = self.dag.nodes.get(last_node_id)
        latest_metrics = current_node.metrics if current_node else env_summary

        # 记录整轮完成日志与最终报告
        GLOBAL_THOUGHT_LOGGER.log_completion(final_clinical_reply, total_elapsed, len(thought_steps))

        # 推送最终完成汇总事件
        yield {
            "type": "complete",
            "reply": final_clinical_reply,
            "action": last_action_name,
            "source": "REACT_VERIFICATION_LOOP",
            "current_version": last_node_id,
            "new_version": last_node_id,
            "layer_name": f"Mask {last_node_id} (质检精细化)",
            "metrics": latest_metrics,
            "elapsed_ms": total_elapsed,
            "thought_steps": thought_steps
        }

    def process_user_instruction(self, user_prompt: str, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """非流式兼容接口 (聚合 stream 输出)"""
        last_result = {}
        for event in self.process_user_instruction_stream(user_prompt, max_iterations=max_iterations):
            if event.get("type") == "complete":
                last_result = event
        return last_result
