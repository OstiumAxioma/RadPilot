import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

class ImageContext:
    """
    医学图像上下文对象，封装当前 3D 图像数据、空间物理属性与当前激活的 Mask
    """
    def __init__(
        self,
        image_data: np.ndarray,
        current_mask: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        affine: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.image_data = image_data
        self.shape = image_data.shape
        self.spacing = tuple(float(s) for s in spacing)  # 各向异性物理体素尺寸 (mm, mm, mm)
        self.voxel_volume_mm3 = float(self.spacing[0] * self.spacing[1] * self.spacing[2])
        self.affine = affine if affine is not None else np.eye(4)
        self.metadata = metadata or {}
        
        if current_mask is not None:
            self.current_mask = (current_mask > 0).astype(np.uint8)
        else:
            self.current_mask = np.zeros(self.shape, dtype=np.uint8)

    def get_summary(self) -> Dict[str, Any]:
        """获取当前环境的物理与几何量化摘要"""
        total_voxels = int(np.count_nonzero(self.current_mask))
        total_volume_cm3 = round((total_voxels * self.voxel_volume_mm3) / 1000.0, 4)
        
        # 计算当前 Mask 的 3D 包围盒 (Bounding Box)
        bbox = None
        if total_voxels > 0:
            indices = np.argwhere(self.current_mask)
            min_idx = indices.min(axis=0).tolist()
            max_idx = indices.max(axis=0).tolist()
            bbox = {"min": min_idx, "max": max_idx}
            
        return {
            "image_dimensions": list(self.shape),
            "voxel_spacing_mm": list(self.spacing),
            "voxel_volume_mm3": round(self.voxel_volume_mm3, 4),
            "mask_total_voxels": total_voxels,
            "mask_volume_cm3": total_volume_cm3,
            "mask_bounding_box": bbox,
            "image_min_intensity": float(np.min(self.image_data)),
            "image_max_intensity": float(np.max(self.image_data)),
            "image_mean_intensity": round(float(np.mean(self.image_data)), 2)
        }

class ToolResult:
    """
    算子执行后的标准返回结构体，强制包含真实物理观测指标 (Observation)
    """
    def __init__(
        self,
        success: bool,
        new_mask: Optional[np.ndarray] = None,
        observation_metrics: Optional[Dict[str, Any]] = None,
        message: str = "",
        error_message: Optional[str] = None,
        action_description: Optional[str] = None,
        attached_image_part: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.new_mask = new_mask if new_mask is not None else np.zeros((0, 0, 0), dtype=np.uint8)
        self.observation_metrics = observation_metrics or {}
        self.message = message or action_description or ("操作成功" if success else "操作失败")
        self.action_description = self.message
        self.error_message = error_message
        self.attached_image_part = attached_image_part

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "observation_metrics": self.observation_metrics,
            "message": self.message,
            "error_message": self.error_message
        }

class BaseMedicalTool(ABC):
    """
    强类型医学图像算子基类
    每个算子必须自描述其名称、功能与 OpenAPI/JSON Schema 参数规范
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一名称，如 'morphological_dilation'"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能描述，供 LLM 进行 Function Calling 规划"""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """参数的 JSON Schema 定义"""
        pass

    @abstractmethod
    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        """执行具体图像计算，返回 ToolResult"""
        pass

    def compute_metrics(
        self,
        old_mask: np.ndarray,
        new_mask: np.ndarray,
        context: ImageContext
    ) -> Dict[str, Any]:
        """
        计算算子执行前后的真实物理与形态学变化指标
        """
        old_count = int(np.count_nonzero(old_mask))
        new_count = int(np.count_nonzero(new_mask))
        diff_count = new_count - old_count
        
        # 计算交集与并集 (Dice / IoU)
        intersection = int(np.count_nonzero(np.logical_and(old_mask > 0, new_mask > 0)))
        union = int(np.count_nonzero(np.logical_or(old_mask > 0, new_mask > 0)))
        dice = round(2.0 * intersection / (old_count + new_count), 4) if (old_count + new_count) > 0 else 1.0
        
        # 物理体积计算 (mm³ 和 cm³)
        volume_change_mm3 = round(diff_count * context.voxel_volume_mm3, 2)
        total_volume_cm3 = round((new_count * context.voxel_volume_mm3) / 1000.0, 4)

        # 3D 包围盒
        bbox = None
        if new_count > 0:
            indices = np.argwhere(new_mask)
            min_idx = indices.min(axis=0).tolist()
            max_idx = indices.max(axis=0).tolist()
            bbox = {"min": min_idx, "max": max_idx}

        # 掩码区域内的原图灰度统计
        mean_intensity = 0.0
        if new_count > 0:
            mean_intensity = round(float(np.mean(context.image_data[new_mask > 0])), 2)

        return {
            "old_voxel_count": old_count,
            "new_voxel_count": new_count,
            "voxel_delta": diff_count,
            "volume_change_mm3": volume_change_mm3,
            "current_volume_cm3": total_volume_cm3,
            "dice_with_previous": dice,
            "bounding_box": bbox,
            "mean_intensity_inside_mask": mean_intensity
        }

    def to_function_declaration(self) -> Dict[str, Any]:
        """导出为 Gemini 标准 Function Calling 声明对象"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }
