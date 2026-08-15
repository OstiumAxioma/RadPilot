import time
import os
import json
import numpy as np
from datetime import datetime

class RadPilotHarness:
    def __init__(self, skills_engine, llm_router):
        self.skills_engine = skills_engine
        self.llm_router = llm_router
        
        # 医生在环状态机: IDLE | PROCESSING | PAUSED_FOR_DOCTOR | COMPLETED
        self.state = "PAUSED_FOR_DOCTOR"
        
        # Mask 版本树 (Lineage Tree)
        self.version_history = []
        self.current_version_index = -1
        
        # 轨迹日志
        self.trajectory_events = []
        
        # 初始化 v0 版本 (空 Mask)
        self._commit_version("v0_init", "INIT", "初始化空白画布与底图", np.zeros(skills_engine.shape, dtype=np.uint8))

    def _commit_version(self, version_tag: str, action_name: str, prompt_text: str, mask_data_3d: np.ndarray):
        """提交一个新的 Mask 版本节点"""
        # 如果在撤销历史中提交新版本，截断 redo 分支
        if self.current_version_index < len(self.version_history) - 1:
            self.version_history = self.version_history[:self.current_version_index + 1]

        version_id = len(self.version_history)
        node = {
            "version_id": f"v{version_id}",
            "tag": version_tag,
            "action": action_name,
            "prompt": prompt_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mask_data": mask_data_3d.copy()
        }
        self.version_history.append(node)
        self.current_version_index = len(self.version_history) - 1
        
        # 同步应用到引擎
        self.skills_engine.set_mask_3d(mask_data_3d)
        
        # 记录轨迹
        self._record_trajectory_event(action_name, prompt_text, f"提交 Mask 版本 v{version_id}")

    def _record_trajectory_event(self, action: str, prompt: str, detail: str):
        """记录轨迹数据"""
        self.trajectory_events.append({
            "step": len(self.trajectory_events) + 1,
            "timestamp": datetime.now().isoformat(),
            "agent_state": self.state,
            "action": action,
            "user_prompt": prompt,
            "detail": detail,
            "current_version": f"v{self.current_version_index}"
        })

    def process_doctor_input(self, prompt: str) -> dict:
        """接收医生自然语言指令，经路由解析后调度 Tool 执行"""
        self.state = "PROCESSING"
        start_time = time.time()
        
        # 1. 意图解析
        intent = self.llm_router.parse_intent(prompt)
        action = intent.get("action", "UNKNOWN")
        explanation = intent.get("explanation", "")
        
        executed = False
        message = ""

        # 2. 调度对应图像 Skill 算子
        if action == "SKULL_STRIP":
            region = intent.get("region", "full")
            new_mask = self.skills_engine.skull_strip_brain_extraction(region=region)
            region_str = "左半脑" if region == "left" else ("右半脑" if region == "right" else "全脑")
            self._commit_version(f"v_skull_strip_{region}", "SKULL_STRIP", prompt, new_mask)
            executed = True
            message = f"已通过空间算子完成{region_str}脑实质分割提取。"

        elif action == "EXPAND":
            pixels = intent.get("pixels", 2)
            new_mask = self.skills_engine.expand_mask(pixels)
            self._commit_version(f"v_expand_{pixels}px", "EXPAND", prompt, new_mask)
            executed = True
            message = f"已将 Mask 边缘外扩 {pixels} 个像素。"

        elif action == "SHRINK":
            pixels = intent.get("pixels", 2)
            new_mask = self.skills_engine.shrink_mask(pixels)
            self._commit_version(f"v_shrink_{pixels}px", "SHRINK", prompt, new_mask)
            executed = True
            message = f"已将 Mask 边缘收缩 {pixels} 个像素。"

        elif action == "REMOVE_ARTIFACTS":
            min_size = intent.get("min_size", 50)
            new_mask = self.skills_engine.remove_artifacts(min_size)
            self._commit_version("v_remove_artifacts", "REMOVE_ARTIFACTS", prompt, new_mask)
            executed = True
            message = "已自动过滤擦除小的离散伪影杂点。"

        elif action == "INVERT":
            new_mask = self.skills_engine.invert_mask()
            self._commit_version("v_invert", "INVERT", prompt, new_mask)
            executed = True
            message = "已翻转反选 Mask。"

        elif action == "RESET":
            new_mask = self.skills_engine.reset_mask()
            self._commit_version("v_reset", "RESET", prompt, new_mask)
            executed = True
            message = "已重置清空 Mask。"

        elif action == "UNDO":
            return self.undo()

        elif action == "REDO":
            return self.redo()

        elif action == "EXPORT":
            export_path = self.export_gold_standard()
            executed = True
            message = f"金标与 Trajectory 已保存完成！导出路径: {export_path}"

        else:
            message = f"未能自动识别该指令或无需算法操作: {explanation}"

        # 切换回挂起在环状态，等待医生下一轮确认或口述
        self.state = "PAUSED_FOR_DOCTOR"
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "success" if executed else "info",
            "action": action,
            "message": message,
            "explanation": explanation,
            "source": intent.get("source", "unknown"),
            "current_version": f"v{self.current_version_index}",
            "elapsed_ms": elapsed_ms,
            "state": self.state
        }

    def undo(self) -> dict:
        """撤销到上一 Mask 版本"""
        if self.current_version_index > 0:
            self.current_version_index -= 1
            prev_node = self.version_history[self.current_version_index]
            self.skills_engine.set_mask_3d(prev_node["mask_data"])
            self.state = "PAUSED_FOR_DOCTOR"
            self._record_trajectory_event("UNDO", "撤销操作", f"回滚到版本 v{self.current_version_index}")
            return {
                "status": "success",
                "action": "UNDO",
                "message": f"已撤销回退至版本 v{self.current_version_index} ({prev_node['action']})",
                "current_version": f"v{self.current_version_index}",
                "state": self.state
            }
        return {
            "status": "error",
            "message": "已经是最初版本，无法继续撤销。",
            "current_version": f"v{self.current_version_index}"
        }

    def redo(self) -> dict:
        """重做至下一 Mask 版本"""
        if self.current_version_index < len(self.version_history) - 1:
            self.current_version_index += 1
            next_node = self.version_history[self.current_version_index]
            self.skills_engine.set_mask_3d(next_node["mask_data"])
            self.state = "PAUSED_FOR_DOCTOR"
            self._record_trajectory_event("REDO", "重做操作", f"前进到版本 v{self.current_version_index}")
            return {
                "status": "success",
                "action": "REDO",
                "message": f"已重做前进至版本 v{self.current_version_index} ({next_node['action']})",
                "current_version": f"v{self.current_version_index}",
                "state": self.state
            }
        return {
            "status": "error",
            "message": "已经是最新版本，无法重做。",
            "current_version": f"v{self.current_version_index}"
        }

    def get_version_tree(self) -> list:
        """获取 Mask 版本线谱摘要列表"""
        return [
            {
                "version_id": item["version_id"],
                "tag": item["tag"],
                "action": item["action"],
                "prompt": item["prompt"],
                "timestamp": item["timestamp"],
                "is_current": (idx == self.current_version_index)
            }
            for idx, item in enumerate(self.version_history)
        ]

    def export_gold_standard(self, output_dir: str = "export") -> str:
        """导出标准 3D NIfTI 金标与轨迹日志"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 导出 3D NIfTI Mask (.nii.gz)
        nii_filename = f"RadPilot_GoldStandard_{timestamp_str}.nii.gz"
        nii_path = os.path.join(output_dir, nii_filename)
        self.skills_engine.export_nifti_mask(nii_path)
        
        # 2. 导出 Trajectory JSON
        traj_filename = f"RadPilot_Trajectory_{timestamp_str}.json"
        traj_path = os.path.join(output_dir, traj_filename)
        
        traj_payload = {
            "session_info": {
                "system": "RadPilot Agent Harness",
                "timestamp": timestamp_str,
                "nifti_source": self.skills_engine.nii_path,
                "gold_standard_export": nii_path,
                "total_steps": len(self.trajectory_events),
                "total_versions": len(self.version_history)
            },
            "trajectory_events": self.trajectory_events,
            "version_lineage": self.get_version_tree()
        }
        
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(traj_payload, f, ensure_ascii=False, indent=2)
            
        print(f"[Harness] 交互轨迹日志已导出至: {traj_path}")
        self.state = "COMPLETED"
        return nii_path
