import numpy as np
from typing import Dict, Any, Optional
from .base_tool import BaseMedicalTool, ImageContext, ToolResult

class IntensityThresholdSegmentationTool(BaseMedicalTool):
    """信号强度与 HU 窗值阈值分割算子"""
    @property
    def name(self) -> str:
        return "segment_by_intensity_range"

    @property
    def description(self) -> str:
        return "按指定的信号强度 (Intensity / HU) 上下限区间进行精确阈值分割。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min_intensity": {
                    "type": "number",
                    "description": "强度下限 (含)。低于该值的体素将被排除。"
                },
                "max_intensity": {
                    "type": "number",
                    "description": "强度上限 (含)。高于该值的体素将被排除。"
                },
                "apply_to_current_mask_only": {
                    "type": "boolean",
                    "description": "是否仅在当前已有 Mask 范围内进行强度过滤 (true 表示交集精细化，false 表示在全图范围内提取)，默认 false。"
                }
            },
            "required": ["min_intensity"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        img = context.image_data
        min_val = kwargs.get("min_intensity", 0.0)
        max_val = kwargs.get("max_intensity", float(np.max(img)))
        apply_within_mask = kwargs.get("apply_to_current_mask_only", False)

        thresholded = np.logical_and(img >= min_val, img <= max_val).astype(np.uint8)

        if apply_within_mask and np.count_nonzero(old_mask) > 0:
            new_mask = np.logical_and(old_mask, thresholded).astype(np.uint8)
            desc = f"在当前 Mask 区域内筛选强度 [{min_val}, {max_val}]"
        else:
            new_mask = thresholded
            desc = f"全局强度区间 [{min_val}, {max_val}] 提取"

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            message=f"已完成 {desc}。提取体素 {metrics['new_voxel_count']} 个，总体积 {metrics['current_volume_cm3']} cm³。"
        )

class ResetMaskTool(BaseMedicalTool):
    """清空/重置当前掩码算子"""
    @property
    def name(self) -> str:
        return "reset_mask"

    @property
    def description(self) -> str:
        return "清空或重置当前工作区的所有分割掩码，回归空白状态。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        new_mask = np.zeros_like(old_mask)
        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            message="已成功重置并清空当前工作区掩码。"
        )
