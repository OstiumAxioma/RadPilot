import os
import json
import time
import requests
import numpy as np
from typing import Dict, List, Any, Optional

from .tools.base_tool import ImageContext, ToolResult
from .tools.registry import GLOBAL_TOOL_REGISTRY, ToolRegistry
from .version_dag import VersionDAG, VersionNode

class AgentEngine:
    """
    RadPilot 专业医学影像 Agent 核心推理引擎
    基于 Gemini Function Calling / Tool Calling 协议，具备真实物理空间感知、多步工具规划与执行观测反馈闭环
    """
    def __init__(
        self,
        image_data: np.ndarray,
        api_key_path: str = "api/gemini_testAPI.txt",
        spacing: tuple = (1.0, 1.0, 1.0),
        tool_registry: Optional[ToolRegistry] = None
    ):
        self.image_data = image_data
        self.spacing = spacing
        self.tool_registry = tool_registry or GLOBAL_TOOL_REGISTRY
        self.dag = VersionDAG(image_data.shape)
        
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
        核心决策执行循环 (Sense -> Plan -> Tool Calling -> Observe -> Report)
        """
        start_time = time.time()
        context = self.get_current_context()
        env_summary = context.get_summary()

        # 处理特殊非工具性快捷指令 (Undo / Redo)
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
                    "elapsed_ms": elapsed
                }
            return {
                "reply": "已处于版本树的初始根节点 (v0)，无法继续撤销。",
                "action": "UNDO_FAILED",
                "source": "VERSION_DAG",
                "current_version": self.dag.current_node_id,
                "elapsed_ms": elapsed
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
                    "elapsed_ms": elapsed
                }
            return {
                "reply": "当前分支已是最新版本节点，没有可重做的后续历史。",
                "action": "REDO_FAILED",
                "source": "VERSION_DAG",
                "current_version": self.dag.current_node_id,
                "elapsed_ms": elapsed
            }

        # 1. 真实发起 Gemini Function Calling 请求
        if not self.api_key:
            return {
                "reply": "未检测到有效的 Gemini API 密钥，请检查 api/gemini_testAPI.txt 配置。",
                "action": "AUTH_ERROR",
                "source": "AGENT_ENGINE",
                "current_version": self.dag.current_node_id,
                "elapsed_ms": int((time.time() - start_time) * 1000)
            }

        system_instruction = (
            "你是一个专业、严谨的放射学智能体 RadPilot。你正在辅助放射科医生对三维医学影像进行解剖学分析与精准分割。\n"
            f"【当前影像物理环境信息】:\n"
            f"- 空间维度 (Shape): {env_summary['image_dimensions']}\n"
            f"- 物理体素间距 (Spacing mm): {env_summary['voxel_spacing_mm']}\n"
            f"- 单个体素物理体积: {env_summary['voxel_volume_mm3']} mm³\n"
            f"- 当前已有 Mask 标定体素量: {env_summary['mask_total_voxels']}\n"
            f"- 当前已有 Mask 标定体积: {env_summary['mask_volume_cm3']} cm³\n"
            f"- 影像信号强度范围: [{env_summary['image_min_intensity']}, {env_summary['image_max_intensity']}], 均值: {env_summary['image_mean_intensity']}\n\n"
            "【行为规范】:\n"
            "1. 仔细分析医生的临床意图。若医生的指令需要进行图像计算（如去颅骨、脑实质提取、外扩、收缩、连通域去噪、阈值分割等），请必须调用对应的 Tool。\n"
            "2. 若医生仅询问影像基本信息或咨询医学问题，请给出专业、简练的放射学解答。\n"
            "3. 绝对不要随意猜测或虚构非工具范围的指令。"
        )

        tools_declarations = self.tool_registry.get_function_declarations()

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
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

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }

        try:
            res = requests.post(self.endpoint, headers=headers, json=payload, timeout=25)
            if res.status_code != 200:
                return {
                    "reply": f"Gemini API 请求失败 (HTTP {res.status_code}): {res.text}",
                    "action": "API_ERROR",
                    "source": "GEMINI_REST",
                    "current_version": self.dag.current_node_id,
                    "elapsed_ms": int((time.time() - start_time) * 1000)
                }

            res_json = res.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                return {
                    "reply": "Gemini 模型未返回有效决策内容。",
                    "action": "EMPTY_CANDIDATES",
                    "source": "GEMINI_REST",
                    "current_version": self.dag.current_node_id,
                    "elapsed_ms": int((time.time() - start_time) * 1000)
                }

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            # 2. 检查模型是否触发了 Function Calling (Tool Calls)
            tool_calls = []
            text_replies = []

            for part in parts:
                if "functionCall" in part:
                    tool_calls.append(part["functionCall"])
                elif "text" in part:
                    text_replies.append(part["text"])

            # 3.1 若无 Tool 调用，直接返回大模型的专业文本答复
            if not tool_calls:
                combined_reply = "\n".join(text_replies).strip()
                return {
                    "reply": combined_reply or "已完成分析，未触发形态学变更。",
                    "action": "CHAT_RESPONSE",
                    "source": "GEMINI_REASONING",
                    "current_version": self.dag.current_node_id,
                    "new_version": self.dag.current_node_id,
                    "elapsed_ms": int((time.time() - start_time) * 1000)
                }

            # 3.2 依次执行工具链，并收集真实物理观察指标 (Observation)
            executed_records = []
            last_result: Optional[ToolResult] = None
            last_tool_name = ""

            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})
                last_tool_name = tool_name

                # 执行工具
                context_snapshot = self.get_current_context()
                result = self.tool_registry.execute_tool(tool_name, context_snapshot, **tool_args)
                last_result = result

                if result.success:
                    # 将执行结果原子化提交到 Version DAG 树
                    node = self.dag.commit(
                        action_name=tool_name.upper(),
                        prompt=user_prompt,
                        new_mask=result.new_mask,
                        metrics=result.observation_metrics,
                        tool_name=tool_name,
                        tool_args=tool_args
                    )
                    executed_records.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "node_id": node.node_id,
                        "metrics": result.observation_metrics,
                        "message": result.message
                    })
                else:
                    executed_records.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "error": result.error_message or result.message
                    })
                    break

            # 4. 第二阶段：将真实的物理观测 (Observation) 回传给 Gemini 生成放射学临床报告
            observations_text = "\n".join([
                f"- 工具 [{r.get('tool')}] 执行结果: {r.get('message', r.get('error'))} (物理观测指标: {json.dumps(r.get('metrics', {}), ensure_ascii=False)})"
                for r in executed_records
            ])

            report_prompt = (
                f"放射科医生指令: \"{user_prompt}\"\n"
                f"底图物理体素间距: {self.spacing} mm\n"
                f"工具流水线执行真实观测反馈 (Real Observations):\n{observations_text}\n\n"
                "请根据以上执行产生的真实物理度量数据，给放射科医生生成一份专业、简练的临床执行结果说明。\n"
                "必须明确汇报：\n"
                "1. 实际执行的算子与参数；\n"
                "2. 掩码体积变化量 (mm³)、当前总标定解剖体积 (cm³)；\n"
                "3. 临床形态学建议（如边缘是否平滑、是否建议进一步去噪或微调）。"
            )

            report_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": report_prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 800
                }
            }

            final_reply = ""
            try:
                report_res = requests.post(self.endpoint, headers=headers, json=report_payload, timeout=20)
                if report_res.status_code == 200:
                    report_json = report_res.json()
                    r_candidates = report_json.get("candidates", [])
                    if r_candidates:
                        final_reply = r_candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            except Exception as e:
                print(f"[AgentEngine Warning] 生成反思报告异常: {e}")

            if not final_reply:
                # 兜底直接组装量化结果
                summary_lines = [r.get("message", "") for r in executed_records if r.get("message")]
                final_reply = "\n".join(summary_lines)

            elapsed = int((time.time() - start_time) * 1000)
            latest_node_id = self.dag.current_node_id
            latest_metrics = executed_records[-1].get("metrics", {}) if executed_records else {}

            return {
                "reply": final_reply,
                "action": last_tool_name.upper() if last_tool_name else "EXECUTE_TOOL",
                "source": "GEMINI_FUNCTION_CALLING",
                "current_version": latest_node_id,
                "new_version": latest_node_id,
                "executed_tools": executed_records,
                "metrics": latest_metrics,
                "elapsed_ms": elapsed
            }

        except Exception as e:
            return {
                "reply": f"Agent 执行发生未捕获异常: {str(e)}",
                "action": "EXCEPTION",
                "source": "AGENT_ENGINE",
                "current_version": self.dag.current_node_id,
                "elapsed_ms": int((time.time() - start_time) * 1000)
            }
