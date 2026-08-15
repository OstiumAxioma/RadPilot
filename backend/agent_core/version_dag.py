import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class VersionNode:
    """
    版本 DAG 树中的原子节点
    包含完整的拓扑关系、工具调用签名、物理指标快照与 Mask 数据
    """
    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        branch_name: str,
        action_name: str,
        prompt: str,
        tool_name: Optional[str],
        tool_args: Dict[str, Any],
        metrics: Dict[str, Any],
        mask_data: np.ndarray
    ):
        self.node_id = node_id
        self.parent_id = parent_id
        self.branch_name = branch_name
        self.action_name = action_name
        self.prompt = prompt
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.metrics = metrics
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mask_data = mask_data.copy()
        self.children_ids: List[str] = []

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "branch_name": self.branch_name,
            "action_name": self.action_name,
            "prompt": self.prompt,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "children_ids": self.children_ids,
            "voxel_count": int(np.count_nonzero(self.mask_data)),
            "metrics": self.metrics
        }

class VersionDAG:
    """
    真有向无环图 (DAG) 临床版本树
    支持多分支分叉 (Branching)、任意节点回滚 (Checkout) 与形态学差异对比 (Diff)
    """
    def __init__(self, initial_shape: Tuple[int, int, int]):
        self.shape = initial_shape
        self.nodes: Dict[str, VersionNode] = {}
        self.current_node_id: str = "v0"
        self.current_branch: str = "main"
        self.branches: Dict[str, str] = {"main": "v0"}
        self._node_counter = 0

        # 初始化 v0 根节点 (空白掩码)
        v0 = VersionNode(
            node_id="v0",
            parent_id=None,
            branch_name="main",
            action_name="INIT",
            prompt="初始化空白工作区",
            tool_name=None,
            tool_args={},
            metrics={"voxel_count": 0, "volume_cm3": 0.0},
            mask_data=np.zeros(initial_shape, dtype=np.uint8)
        )
        self.nodes["v0"] = v0

    def commit(
        self,
        action_name: str,
        prompt: str,
        new_mask: np.ndarray,
        metrics: Dict[str, Any],
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None
    ) -> VersionNode:
        """
        提交新版本节点，挂载在当前节点 (parent) 下。若在中间节点提交，则自然形成新分支，绝不截断历史！
        """
        self._node_counter += 1
        new_id = f"v{self._node_counter}"
        
        parent_node = self.nodes.get(self.current_node_id)
        parent_id = parent_node.node_id if parent_node else None

        # 检查分支名：如果当前父节点已有其他子节点，自动生成分叉分支名
        branch = self.current_branch
        if parent_node and len(parent_node.children_ids) > 0:
            branch = f"branch_{new_id}"
            self.current_branch = branch

        node = VersionNode(
            node_id=new_id,
            parent_id=parent_id,
            branch_name=branch,
            action_name=action_name,
            prompt=prompt,
            tool_name=tool_name,
            tool_args=tool_args or {},
            metrics=metrics,
            mask_data=new_mask
        )

        self.nodes[new_id] = node
        if parent_node:
            parent_node.children_ids.append(new_id)

        self.current_node_id = new_id
        return node

    def create_branch(self, branch_name: str):
        """显式创建并切换到指定名称的新分支"""
        self.current_branch = branch_name
        self.branches[branch_name] = self.current_node_id

    def checkout(self, node_id: str) -> Optional[VersionNode]:
        """
        检出并切换工作区到指定历史节点 (可从任意历史节点继续演进，不破坏原有分支)
        """
        if node_id in self.nodes:
            self.current_node_id = node_id
            self.current_branch = self.nodes[node_id].branch_name
            return self.nodes[node_id]
        return None

    def undo(self) -> Optional[VersionNode]:
        """撤销到父节点"""
        curr = self.nodes.get(self.current_node_id)
        if curr and curr.parent_id and curr.parent_id in self.nodes:
            return self.checkout(curr.parent_id)
        return None

    def redo(self) -> Optional[VersionNode]:
        """重做到最近的子节点"""
        curr = self.nodes.get(self.current_node_id)
        if curr and len(curr.children_ids) > 0:
            # 默认走向最新的子分支
            target_id = curr.children_ids[-1]
            return self.checkout(target_id)
        return None

    def get_current_mask(self) -> np.ndarray:
        """获取当前激活节点的 3D 掩码数据"""
        curr = self.nodes.get(self.current_node_id)
        if curr is not None:
            return curr.mask_data
        return np.zeros(self.shape, dtype=np.uint8)

    def get_tree_structure(self) -> List[Dict[str, Any]]:
        """导出完整的树结构供前端展示版本图谱"""
        return [node.to_summary_dict() for node in self.nodes.values()]

    def compute_diff(self, node_id_a: str, node_id_b: str, voxel_volume_mm3: float = 1.0) -> Dict[str, Any]:
        """计算两个任意版本节点之间的形态学差异与 Dice 相似度"""
        if node_id_a not in self.nodes or node_id_b not in self.nodes:
            return {"error": "NODE_NOT_FOUND"}

        mask_a = self.nodes[node_id_a].mask_data > 0
        mask_b = self.nodes[node_id_b].mask_data > 0

        count_a = int(np.count_nonzero(mask_a))
        count_b = int(np.count_nonzero(mask_b))
        intersection = int(np.count_nonzero(np.logical_and(mask_a, mask_b)))
        union = int(np.count_nonzero(np.logical_or(mask_a, mask_b)))

        dice = round(2.0 * intersection / (count_a + count_b), 4) if (count_a + count_b) > 0 else 1.0
        iou = round(intersection / union, 4) if union > 0 else 1.0

        diff_voxels = count_b - count_a
        diff_volume_mm3 = round(diff_voxels * voxel_volume_mm3, 2)

        return {
            "node_a": node_id_a,
            "node_b": node_id_b,
            "voxel_count_a": count_a,
            "voxel_count_b": count_b,
            "intersection_voxels": intersection,
            "dice_similarity": dice,
            "dice_score": dice,
            "iou": iou,
            "voxel_difference": diff_voxels,
            "volume_difference_mm3": diff_volume_mm3
        }
