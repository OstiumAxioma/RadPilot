import os
import json
import time
import requests
import numpy as np
from scipy import ndimage
from typing import Dict, List, Any, Optional

from .tools.base_tool import ImageContext, ToolResult
from .tools.registry import GLOBAL_TOOL_REGISTRY, ToolRegistry
from .version_dag import VersionDAG, VersionNode
from .multimodal.slice_encoder import MultiModalSliceEncoder

class AgentEngine:
    """
    RadPilot 专业医学影像 ReAct 自主推理引擎
    具备【执行-质检验收-反思精修 (Execution -> Verification Gate -> Refinement)】闭环状态机
    """
    def __init__(
        self,
        image_data: np.ndarray,
        api_key_path: str = "api/gemini_testAPI.txt",
        spacing: tuple = (1.0, 1.0, 1.0),
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 4
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

    def process_user_instruction(self, user_prompt: str) -> Dict[str, Any]:
        """
        核心 ReAct 自主精修与质检验收闭环循环 (Execution & Verification Gate Loop)
        """
        start_time = time.time()
        context = self.get_current_context()
        env_summary = context.get_summary()

        # 处理快捷指令 (Undo / Redo)
        prompt_lower = user_prompt.strip().lower()
        if prompt_lower in ["撤销", "undo", "上一步", "后退"]:
            prev_node = self.dag.undo()
            elapsed = int((time.time() - start_time) * 1000)
            if prev_node:
                return {
                    "reply": f"已成功撤销至历史版本 [{prev_node.node_id}] ({prev_node.action_name})。当前标定体积: {prev_node.metrics.get('current_volume_cm3', 0)} cm³。",
                    "action": "UNDO",
                    "source": "VERSION_DAG",
                    "current_version": prev_node.node_id,
                    "new_version": prev_node.node_id,
                    "metrics": prev_node.metrics,
                    "elapsed_ms": elapsed,
                    "thought_steps": []
                }
            return {
                "reply": "已处于版本树的初始根节点 (v0)，无法继续撤销。",
                "action": "UNDO_FAILED",
                "source": "VERSION_DAG",
                "current_version": self.dag.current_node_id,
                "elapsed_ms": elapsed,
                "thought_steps": []
            }

        if prompt_lower in ["重做", "redo", "下一步", "前进"]:
            next_node = self.dag.redo()
            elapsed = int((time.time() - start_time) * 1000)
            if next_node:
                return {
                    "reply": f"已成功重做至版本分支 [{next_node.node_id}] ({next_node.action_name})。当前标定体积: {next_node.metrics.get('current_volume_cm3', 0)} cm³。",
                    "action": "REDO",
                    "source": "VERSION_DAG",
                    "current_version": next_node.node_id,
                    "new_version": next_node.node_id,
                    "metrics": next_node.metrics,
                    "elapsed_ms": elapsed,
                    "thought_steps": []
                }
            return {
                "reply": "当前分支已是最新版本节点，没有可重做的后续历史。",
                "action": "REDO_FAILED",
                "source": "VERSION_DAG",
                "current_version": self.dag.current_node_id,
                "elapsed_ms": elapsed,
                "thought_steps": []
            }

        if not self.api_key:
            return {
                "reply": "未检测到有效的 Gemini API 密钥，请检查 api/gemini_testAPI.txt 配置。",
                "action": "AUTH_ERROR",
                "source": "AGENT_ENGINE",
                "current_version": self.dag.current_node_id,
                "elapsed_ms": int((time.time() - start_time) * 1000),
                "thought_steps": []
            }

        system_instruction = (
            "你是一个具备原生多模态视觉空间推理与【执行-验收-精修】严谨闭环的资深放射学智能体 RadPilot。\n"
            "【多模态视觉输入说明】:\n"
            "系统已为你提供了当前三维医学体数据的正交三视角中心断层画廊与坐标标尺:\n"
            "1. 【图1: 轴位 6 格画廊】: 从颅底小脑(Z=18%)、第四脑室(Z=30%)到大脑半卵圆中心(Z=68%)、顶叶(Z=82%)\n"
            "2. 【图2: 冠状位 3 格画廊】: 前额叶(Y=30%)、中线脑干(Y=50%)、后颅窝小脑/枕叶(Y=70%)\n"
            "3. 【图3: 矢状位 3 格画廊】: 右半球(X=30%)、正中矢状面(X=50%)、左半球(X=70%)\n\n"
            f"【物理空间元数据】:\n"
            f"- 空间维度 (Shape: X, Y, Z): {env_summary['image_dimensions']}\n"
            f"- 物理体素间距 (Spacing mm): {env_summary['voxel_spacing_mm']}\n"
            f"- 单个体素物理体积: {env_summary['voxel_volume_mm3']} mm³\n"
            f"- 当前已有 Mask 标定体积: {env_summary['mask_volume_cm3']} cm³\n"
            f"- 影像信号强度范围: [{env_summary['image_min_intensity']}, {env_summary['image_max_intensity']}], 均值: {env_summary['image_mean_intensity']}\n\n"
            "【必须严格遵循的 ReAct 执行与质检验收流程 (Verification Gate Loop)】:\n"
            "1. 【阶段一: 初始提取】: 调用 `extract_brain_tissue`、`spatial_prompt_guided_segmentation` 或 `threshold_range` 等工具完成初始掩码生成；\n"
            "2. 【阶段二: 质量审查与反思验收 (Quality Inspection)】: 系统在每个工具执行后都会重新截取叠加上 Mask 的最新画廊切片并计算连通域孤岛数。你必须审视最新切片与指标，若发现孤立杂质或边缘孔洞，必须继续调用 `island_and_smooth` 或 `erase_brush_3d`/`scissors_cut` 进行二次精修；\n"
            "3. 【阶段三: 终审报告】: 只有在确认掩码解剖准确、无多余外皮、无孤岛碎屑时，方可输出最终临床定量报告。\n"
            "在调用每个工具前，必须显式输出一段【Thought】说明诊断思考与工具调用动机。"
        )

        tools_declarations = self.tool_registry.get_function_declarations()

        # 初始动态截取切片图像帧
        multiview_image_parts = MultiModalSliceEncoder.encode_multiview_slices(
            self.image_data,
            current_mask=context.current_mask
        )

        # 构建对话历史
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

        # -------------------------------------------------------------
        # 核心 ReAct 状态机循环 (执行 -> 质检反馈 -> 再精修)
        # -------------------------------------------------------------
        for iteration in range(1, self.max_iterations + 1):
            step_start = time.time()
            
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
                res = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
                if res.status_code != 200:
                    error_msg = f"Gemini API 请求失败 (HTTP {res.status_code}): {res.text}"
                    print(f"[ReAct Loop Error] {error_msg}")
                    final_clinical_reply = error_msg
                    break

                res_json = res.json()
                candidates = res_json.get("candidates", [])
                if not candidates:
                    final_clinical_reply = "模型未返回有效候选内容。"
                    break

                candidate_content = candidates[0].get("content", {})
                parts = candidate_content.get("parts", [])

                # 解析本轮输出的 Thought 与 Tool Calls
                tool_calls = []
                text_chunks = []
                for p in parts:
                    if "functionCall" in p:
                        tool_calls.append(p["functionCall"])
                    elif "text" in p:
                        text_chunks.append(p["text"])

                current_thought = "\n".join(text_chunks).strip()

                # 如果模型没有触发工具调用：说明模型认为质检完全通过，输出终审报告
                if not tool_calls:
                    final_clinical_reply = current_thought or "已完成解剖分割并通过质检验收。"
                    break

                # 依次执行工具下发
                for call in tool_calls:
                    tool_name = call.get("name")
                    tool_args = call.get("args", {})
                    last_action_name = tool_name.upper()

                    # 执行工具
                    current_ctx = self.get_current_context()
                    tool_res = self.tool_registry.execute_tool(tool_name, current_ctx, **tool_args)
                    
                    if tool_res.success:
                        # 提交 DAG 版本节点
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

                    # 计算质控指标 (连通分支数、孔洞数等)
                    curr_mask = self.dag.get_current_mask()
                    labeled_islands, island_count = ndimage.label(curr_mask > 0)
                    
                    obs_summary = (
                        f"已执行 {tool_res.message}。"
                        f"变化体积: {tool_res.observation_metrics.get('volume_change_mm3', 0)} mm³，"
                        f"当前标定总体积: {tool_res.observation_metrics.get('current_volume_cm3', 0)} cm³。"
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
                        "elapsed_ms": step_elapsed
                    }
                    thought_steps.append(step_info)

                    # -------------------------------------------------------------
                    # 【核心验收门控 (Verification Gate)】:
                    # 全方位进行过分割 (Over-segmentation)、欠分割 (Under-segmentation) 与解剖中线越界自检
                    # -------------------------------------------------------------
                    curr_mask = self.dag.get_current_mask()
                    labeled_islands, island_count = ndimage.label(curr_mask > 0)
                    current_vol_cm3 = tool_res.observation_metrics.get('current_volume_cm3', 0.0)
                    dim_x, dim_y, dim_z = context.shape
                    mid_x = dim_x // 2

                    # 1. 深度过分割与越界自检
                    leakage_warnings = []
                    
                    # 检查 1: 单侧大脑半球中线越界检查
                    if any(k in user_prompt for k in ["左脑", "左半球", "left"]):
                        overflow_right = int(np.count_nonzero(curr_mask[:mid_x, :, :] > 0))
                        if overflow_right > 50:
                            leakage_warnings.append(f"【中线过分割越界】: 检测到掩码越过正中矢状面侵入右脑 {overflow_right} 个体素，必须调用 `scissors_cut(plane='sagittal', cut_index={mid_x}, remove_side='less_than')` 切除！")
                        if current_vol_cm3 > 750.0:
                            leakage_warnings.append(f"【生理容积过分割】: 当前体积 ({current_vol_cm3} cm³) 显著超出成年人单侧半球正常生理范围 (500~700 cm³)，疑似包入了颅底软组织或对侧脑，必须使用 `scissors_cut` 或 `erase_brush_3d` 剔除多余部分！")

                    elif any(k in user_prompt for k in ["右脑", "右半球", "right"]):
                        overflow_left = int(np.count_nonzero(curr_mask[mid_x:, :, :] > 0))
                        if overflow_left > 50:
                            leakage_warnings.append(f"【中线过分割越界】: 检测到掩码越过正中矢状面侵入左脑 {overflow_left} 个体素，必须调用 `scissors_cut(plane='sagittal', cut_index={mid_x}, remove_side='greater_than')` 切除！")
                        if current_vol_cm3 > 750.0:
                            leakage_warnings.append(f"【生理容积过分割】: 当前体积 ({current_vol_cm3} cm³) 显著超出成年人单侧半球正常生理范围 (500~700 cm³)，必须修剪！")

                    elif any(k in user_prompt for k in ["小脑", "cerebellum"]):
                        overflow_z = int(np.count_nonzero(curr_mask[:, :, int(dim_z * 0.42):] > 0))
                        if overflow_z > 50:
                            leakage_warnings.append(f"【向上过分割】: 小脑掩码向上溢出侵入枕叶 ({overflow_z} 个体素)，必须调用 `scissors_cut(plane='axial', cut_index={int(dim_z*0.40)}, remove_side='greater_than')` 切除！")
                        if current_vol_cm3 > 200.0:
                            leakage_warnings.append(f"【小脑生理容积过分割】: 当前体积 ({current_vol_cm3} cm³) 超过正常成人小脑范围 (120~180 cm³)，存在过度包绕！")

                    # 检查 2: 独立孤岛碎屑检查
                    if island_count > 1:
                        leakage_warnings.append(f"【孤立碎屑伪影】: 存在 {island_count} 个独立不连通的离散孤岛，建议调用 `island_and_smooth` 过滤杂质。")

                    # 重新截取叠加了最新 Mask 的画廊图像
                    overlay_slices = MultiModalSliceEncoder.encode_multiview_slices(
                        self.image_data,
                        current_mask=curr_mask
                    )

                    warning_text = "\n".join([f"- ⚠️ {w}" for w in leakage_warnings]) if leakage_warnings else "- ✅ 各项解剖边界与生理容积指标质检正常，未检测到明显过分割或中线泄漏。"

                    inspection_prompt = (
                        f"【系统质检验收与过分割自检反馈 (Iteration {iteration})】:\n"
                        f"- 工具 [{tool_name}] 已执行完毕。\n"
                        f"- 当前掩码物理体积: {current_vol_cm3} cm³\n"
                        f"- 独立连通孤岛分支数: {island_count} 个\n"
                        f"{warning_text}\n\n"
                        "请审视附带的最新叠加掩码画廊切片进行验收审查：\n"
                        "1. 【过分割判定】: 若存在过分割 (超出解剖边界、侵入对侧半球、包入颅外软组织)，必须调用 `scissors_cut`、`erase_brush_3d` 或 `morphological_erosion` 进行修剪与断桥；\n"
                        "2. 【欠分割/孤岛判定】: 若边缘不光滑或存在离散碎屑，调用 `island_and_smooth`；\n"
                        "3. 【终审合格】: 只有在确认掩码解剖完全严密、无过分割、无孤岛碎屑时，方可输出最终临床评估总结。"
                    )

                    # 追加到对话历史，驱动大模型进入下一轮 Loop
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
                        ] + overlay_slices + [{"text": inspection_prompt}]
                    })

            except Exception as e:
                import traceback
                print(f"[ReAct Loop Exception]: {traceback.format_exc()}")
                final_clinical_reply = f"ReAct 推理循环异常: {str(e)}"
                break

        if not final_clinical_reply:
            final_clinical_reply = f"已通过 ReAct 验收闭环完成精细化勾画，共执行 {len(thought_steps)} 步迭代。"

        total_elapsed = int((time.time() - start_time) * 1000)
        current_node = self.dag.nodes.get(last_node_id)
        latest_metrics = current_node.metrics if current_node else env_summary

        return {
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
