import os
import json
import time
import datetime
import numpy as np
from typing import Dict, Any, Optional

class NumpyJSONEncoder(json.JSONEncoder):
    """支持 NumPy 标量、数组及各类特殊对象的安全 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(obj, cls=NumpyJSONEncoder, ensure_ascii=False, **kwargs)

class AgentThoughtLogger:
    """
    RadPilot 临床思维链与模型决策日志记录器
    自动将每一轮对话、Gemini 原始 Payload、Thought 诊断思维、Tool 调用参数、
    Observation 真实物理指标以及质检验收结果持久化至 backend/logs/ 目录。
    """
    def __init__(self, log_dir: Optional[str] = None):
        if not log_dir:
            # 默认存放在 backend/logs 目录下
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.log_dir = os.path.join(base_dir, "backend", "logs")
        else:
            self.log_dir = log_dir
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"agent_cot_{self.session_id}.log")
        self.jsonl_file = os.path.join(self.log_dir, f"agent_cot_{self.session_id}.jsonl")
        
        self._init_log_file()

    def _init_log_file(self):
        header = (
            f"================================================================================\n"
            f"  RadPilot 放射学智能体 ReAct 思维链与模型输出排查日志\n"
            f"  会话时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  日志文件: {self.log_file}\n"
            f"================================================================================\n\n"
        )
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(header)
        except Exception as e:
            print(f"[Logger Error] 无法初始化日志文件: {e}")

    def log_interaction_start(self, user_prompt: str, env_summary: Dict[str, Any]):
        """记录一次医生自然语言交互的启动"""
        time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        text = (
            f"\n{'#'*80}\n"
            f"[{time_str}] 🧑‍⚕️ 医生输入指令: \"{user_prompt}\"\n"
            f"空间环境: Dimensions={env_summary.get('image_dimensions')}, Spacing={env_summary.get('voxel_spacing_mm')}, "
            f"当前Mask体积={env_summary.get('mask_volume_cm3')} cm³\n"
            f"{'-'*80}\n"
        )
        self._write_text(text)

    def log_step(
        self,
        iteration: int,
        thought: str,
        tool_calls: list,
        tool_results: list,
        verification_feedback: Optional[str] = None,
        elapsed_ms: int = 0
    ):
        """记录 ReAct 单轮迭代中的完整思维链 (Thought -> Action -> Observation -> Verification)"""
        time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        lines = [
            f"\n▶ [{time_str}] [Iteration {iteration}] (耗时: {elapsed_ms}ms)",
            f"  🧠 【Thought / 临床诊断思考】:"
        ]
        
        if thought:
            for t_line in thought.strip().splitlines():
                lines.append(f"     {t_line}")
        else:
            lines.append("     (模型未输出显式文本，直接触发工具调用)")

        if tool_calls:
            lines.append("  ⚡ 【Action / 工具下发调用】:")
            for call in tool_calls:
                t_name = call.get("name")
                t_args = call.get("args", {})
                lines.append(f"     - 算子: {t_name}")
                lines.append(f"       参数: {safe_json_dumps(t_args, indent=2)}")

        if tool_results:
            lines.append("  📊 【Observation / 底层物理观察指标】:")
            for res in tool_results:
                lines.append(f"     - 结果: {res.get('message')}")
                lines.append(f"       物理度量: {safe_json_dumps(res.get('metrics', {}))}")

        if verification_feedback:
            lines.append("  🛡️ 【Verification Gate / 系统质检验收反馈】:")
            for v_line in verification_feedback.strip().splitlines():
                lines.append(f"     {v_line}")

        lines.append(f"  {'.' * 60}")
        self._write_text("\n".join(lines) + "\n")

        # 同步写入结构化 JSONL 方便自动化解析
        json_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "iteration": iteration,
            "thought": thought,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "verification_feedback": verification_feedback,
            "elapsed_ms": elapsed_ms
        }
        self._write_jsonl(json_record)

    def log_completion(self, final_reply: str, total_elapsed_ms: int, total_steps: int):
        """记录整轮 ReAct 结束与最终报告"""
        time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        text = (
            f"\n🏁 [{time_str}] 【ReAct 质检闭环完成】 (总步数: {total_steps}, 总耗时: {total_elapsed_ms}ms)\n"
            f"📋 最终临床总结报告:\n"
            f"{final_reply}\n"
            f"{'#'*80}\n\n"
        )
        self._write_text(text)

    def _write_text(self, text: str):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[Logger Error] 写入文本日志失败: {e}")

    def _write_jsonl(self, data: dict):
        try:
            with open(self.jsonl_file, "a", encoding="utf-8") as f:
                f.write(safe_json_dumps(data) + "\n")
        except Exception as e:
            print(f"[Logger Error] 写入 JSONL 失败: {e}")

# 全局单例日志对象
GLOBAL_THOUGHT_LOGGER = AgentThoughtLogger()
