import numpy as np
from scipy import ndimage
from typing import Dict, Any, Optional
from .base_tool import BaseMedicalTool, ImageContext, ToolResult

def _generate_anisotropic_struct_element(spacing: tuple, radius_mm: float) -> np.ndarray:
    """
    根据三维各向异性物理间距 (Spacing) 生成真实椭球形态学结构元素
    """
    rx = max(1, int(round(radius_mm / spacing[0])))
    ry = max(1, int(round(radius_mm / spacing[1])))
    rz = max(1, int(round(radius_mm / spacing[2])))
    
    x, y, z = np.ogrid[-rx:rx+1, -ry:ry+1, -rz:rz+1]
    struct = ((x * spacing[0])**2 + (y * spacing[1])**2 + (z * spacing[2])**2) <= (radius_mm**2)
    return struct.astype(bool)

class MorphologicalDilationTool(BaseMedicalTool):
    """三维形态学外扩 / 膨胀算子"""
    @property
    def name(self) -> str:
        return "morphological_dilation"

    @property
    def description(self) -> str:
        return "对当前激活的 3D 掩码进行形态学向外扩张 (膨胀)。支持指定物理半径 (毫米 mm) 或体素步长 (pixels)。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "radius_mm": {
                    "type": "number",
                    "description": "向外扩张的物理半径 (单位: 毫米 mm)。若未指定则由 pixels 换算，默认为 2.0mm。"
                },
                "pixels": {
                    "type": "integer",
                    "description": "向外扩张的体素步长 (voxels/pixels)。可选参数，如 1, 2, 3。"
                }
            },
            "required": []
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        if np.count_nonzero(old_mask) == 0:
            return ToolResult(
                success=False,
                new_mask=old_mask,
                observation_metrics=self.compute_metrics(old_mask, old_mask, context),
                message="当前工作区中没有已激活的 Mask，无法执行膨胀操作。请先执行分割或提取算子。",
                error_message="EMPTY_MASK_PRECONDITION"
            )

        radius_mm = kwargs.get("radius_mm")
        pixels = kwargs.get("pixels")

        if radius_mm is not None and radius_mm > 0:
            struct = _generate_anisotropic_struct_element(context.spacing, float(radius_mm))
            iterations = 1
            desc = f"物理半径 {radius_mm}mm"
        elif pixels is not None and pixels > 0:
            struct = ndimage.generate_binary_structure(3, 1)
            iterations = int(pixels)
            desc = f"{pixels} 体素步长"
        else:
            # 默认 2.0 毫米
            struct = _generate_anisotropic_struct_element(context.spacing, 2.0)
            iterations = 1
            desc = "默认 2.0mm 物理半径"

        new_mask = ndimage.binary_dilation(old_mask, structure=struct, iterations=iterations).astype(np.uint8)
        metrics = self.compute_metrics(old_mask, new_mask, context)

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            message=f"已成功执行形态学外扩 ({desc})。体积增加 {metrics['volume_change_mm3']} mm³，当前总标定体积为 {metrics['current_volume_cm3']} cm³。"
        )

class MorphologicalErosionTool(BaseMedicalTool):
    """三维形态学收缩 / 腐蚀算子"""
    @property
    def name(self) -> str:
        return "morphological_erosion"

    @property
    def description(self) -> str:
        return "对当前激活的 3D 掩码进行形态学向内收缩 (腐蚀)。支持指定物理半径 (毫米 mm) 或体素步长 (pixels)。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "radius_mm": {
                    "type": "number",
                    "description": "向内收缩的物理半径 (单位: 毫米 mm)。"
                },
                "pixels": {
                    "type": "integer",
                    "description": "向内收缩的体素步长 (voxels/pixels)。"
                }
            },
            "required": []
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        if np.count_nonzero(old_mask) == 0:
            return ToolResult(
                success=False,
                new_mask=old_mask,
                observation_metrics=self.compute_metrics(old_mask, old_mask, context),
                message="当前工作区中没有已激活的 Mask，无法执行腐蚀操作。",
                error_message="EMPTY_MASK_PRECONDITION"
            )

        radius_mm = kwargs.get("radius_mm")
        pixels = kwargs.get("pixels")

        if radius_mm is not None and radius_mm > 0:
            struct = _generate_anisotropic_struct_element(context.spacing, float(radius_mm))
            iterations = 1
            desc = f"物理半径 {radius_mm}mm"
        elif pixels is not None and pixels > 0:
            struct = ndimage.generate_binary_structure(3, 1)
            iterations = int(pixels)
            desc = f"{pixels} 体素步长"
        else:
            struct = _generate_anisotropic_struct_element(context.spacing, 2.0)
            iterations = 1
            desc = "默认 2.0mm 物理半径"

        new_mask = ndimage.binary_erosion(old_mask, structure=struct, iterations=iterations).astype(np.uint8)
        metrics = self.compute_metrics(old_mask, new_mask, context)

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            message=f"已成功执行形态学收缩 ({desc})。体积减少 {abs(metrics['volume_change_mm3'])} mm³，当前总标定体积为 {metrics['current_volume_cm3']} cm³。"
        )

class ConnectedComponentFilterTool(BaseMedicalTool):
    """三维连通域分析与孤立碎屑/伪影过滤算子"""
    @property
    def name(self) -> str:
        return "filter_connected_components"

    @property
    def description(self) -> str:
        return "三维连通域分析去噪。可按最小物理体积阈值 (min_volume_mm3) 剔除微小噪点伪影，或仅保留最大的 K 个主要解剖结构 (keep_top_k)。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min_volume_mm3": {
                    "type": "number",
                    "description": "过滤的最小物理体积阈值 (单位: mm³)。小于该体积的孤立碎片将被移除，默认 50.0mm³。"
                },
                "keep_top_k": {
                    "type": "integer",
                    "description": "仅保留体积最大的前 K 个连通域分支 (例如 1 表示只保留最大单一主体)。"
                }
            },
            "required": []
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        if np.count_nonzero(old_mask) == 0:
            return ToolResult(
                success=False,
                new_mask=old_mask,
                observation_metrics=self.compute_metrics(old_mask, old_mask, context),
                message="当前工作区中没有已激活的 Mask，无需去噪。",
                error_message="EMPTY_MASK_PRECONDITION"
            )

        labeled_array, num_features = ndimage.label(old_mask)
        if num_features == 0:
            return ToolResult(
                success=True,
                new_mask=old_mask,
                observation_metrics=self.compute_metrics(old_mask, old_mask, context),
                message="未检测到独立连通域。"
            )

        sizes = ndimage.sum(old_mask, labeled_array, range(1, num_features + 1))
        min_volume_mm3 = kwargs.get("min_volume_mm3", 50.0)
        min_voxels = max(1, int(round(min_volume_mm3 / context.voxel_volume_mm3)))
        keep_top_k = kwargs.get("keep_top_k")

        new_mask = np.zeros_like(old_mask)

        if keep_top_k is not None and keep_top_k > 0:
            sorted_indices = np.argsort(sizes)[::-1]
            top_labels = [i + 1 for i in sorted_indices[:int(keep_top_k)]]
            for lab in top_labels:
                new_mask[labeled_array == lab] = 1
            desc = f"保留体积最大的前 {keep_top_k} 个主要解剖连通域"
        else:
            valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
            for lab in valid_labels:
                new_mask[labeled_array == lab] = 1
            desc = f"过滤小于 {min_volume_mm3} mm³ 的微小碎屑伪影 (共过滤 {num_features - len(valid_labels)} 个噪点)"

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            message=f"已完成连通域去噪 ({desc})。移除体素 {abs(metrics['voxel_delta'])} 个，清理体积 {abs(metrics['volume_change_mm3'])} mm³。"
        )

class MorphologicalSmoothTool(BaseMedicalTool):
    """三维形态学平滑与孔洞充填算子"""
    @property
    def name(self) -> str:
        return "morphological_smooth"

    @property
    def description(self) -> str:
        return "利用三维闭运算与孔洞填充平滑掩码边界，修复内部孔洞与边缘粗糙凹陷。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fill_holes": {
                    "type": "boolean",
                    "description": "是否对三维内部封闭孔洞进行完全充填，默认为 true。"
                },
                "smooth_radius_mm": {
                    "type": "number",
                    "description": "闭运算平滑半径 (毫米 mm)，默认为 1.5mm。"
                }
            },
            "required": []
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        if np.count_nonzero(old_mask) == 0:
            return ToolResult(
                success=False,
                new_mask=old_mask,
                observation_metrics=self.compute_metrics(old_mask, old_mask, context),
                message="当前工作区中没有已激活的 Mask，无法平滑。",
                error_message="EMPTY_MASK_PRECONDITION"
            )

        radius_mm = kwargs.get("smooth_radius_mm", 1.5)
        fill_holes = kwargs.get("fill_holes", True)

        struct = _generate_anisotropic_struct_element(context.spacing, float(radius_mm))
        # 闭运算: 膨胀后腐蚀，填补狭缝
        closed_mask = ndimage.binary_closing(old_mask, structure=struct).astype(np.uint8)

        if fill_holes:
            new_mask = ndimage.binary_fill_holes(closed_mask).astype(np.uint8)
        else:
            new_mask = closed_mask

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            message=f"已完成形态学边界平滑与孔洞充填。修复体积 {metrics['volume_change_mm3']} mm³。"
        )
