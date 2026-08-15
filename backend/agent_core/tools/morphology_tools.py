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

class AnalyzeConnectivityTool(BaseMedicalTool):
    """【主动连通性深度检测与诊断探针算子 (Analyze Mask Connectivity)】"""
    @property
    def name(self) -> str:
        return "analyze_connectivity"

    @property
    def description(self) -> str:
        return (
            "【主动诊断探针: 连通性深度分析 (Analyze Connectivity)】对当前 3D 掩码进行拓扑连通性扫描。"
            "返回所有独立连通域 (Islands) 的详细解剖档案: 包含每个孤岛的体积 (cm³)、三维质心坐标 [X,Y,Z]、包围盒 BBox、三视角切片跨度 (Z/Y/X Span) 及信号均值。"
            "帮助智能体精确发现哪些是主解剖器官，哪些是离散粘连碎片，并直接获取碎片的坐标以便精准擦除或过滤！"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_reported_islands": {
                    "type": "integer",
                    "description": "最大汇报的前 N 个连通域档案 (默认 10 个)。"
                },
                "min_debris_volume_mm3": {
                    "type": "number",
                    "description": "认定为碎屑的体积上限 (mm³，默认 500.0)。"
                }
            }
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        img = context.image_data.astype(np.float32)
        dim_x, dim_y, dim_z = context.shape
        voxel_vol = context.voxel_volume_mm3
        max_n = int(kwargs.get("max_reported_islands", 10))
        debris_threshold_mm3 = float(kwargs.get("min_debris_volume_mm3", 500.0))

        if np.count_nonzero(old_mask) == 0:
            return ToolResult(
                success=True,
                new_mask=old_mask,
                observation_metrics={"total_islands": 0, "current_volume_cm3": 0.0},
                message="当前掩码为空 (0 体素)，连通域数量为 0。"
            )

        labeled, num_features = ndimage.label(old_mask)
        total_voxels = np.count_nonzero(old_mask)
        total_vol_cm3 = round(total_voxels * voxel_vol / 1000.0, 3)

        sizes = ndimage.sum(old_mask, labeled, range(1, num_features + 1))
        sorted_indices = np.argsort(sizes)[::-1]

        islands_list = []
        debris_count = 0

        for rank, idx in enumerate(sorted_indices[:max_n]):
            lab = idx + 1
            v_count = int(sizes[idx])
            v_vol_mm3 = v_count * voxel_vol
            v_vol_cm3 = round(v_vol_mm3 / 1000.0, 3)
            ratio_pct = round((v_count / total_voxels) * 100.0, 2)

            if v_vol_mm3 <= debris_threshold_mm3:
                debris_count += 1

            island_mask = (labeled == lab)
            coords = np.argwhere(island_mask)
            if len(coords) > 0:
                cx = int(round(np.mean(coords[:, 0])))
                cy = int(round(np.mean(coords[:, 1])))
                cz = int(round(np.mean(coords[:, 2])))

                min_x, min_y, min_z = int(np.min(coords[:, 0])), int(np.min(coords[:, 1])), int(np.min(coords[:, 2]))
                max_x, max_y, max_z = int(np.max(coords[:, 0])), int(np.max(coords[:, 1])), int(np.max(coords[:, 2]))

                mean_val = round(float(np.mean(img[island_mask])), 1)

                islands_list.append({
                    "island_id": rank + 1,
                    "label_index": int(lab),
                    "volume_cm3": v_vol_cm3,
                    "volume_mm3": round(v_vol_mm3, 1),
                    "voxel_count": v_count,
                    "volume_ratio_pct": ratio_pct,
                    "centroid_3d": [cx, cy, cz],
                    "bbox_3d": [min_x, min_y, min_z, max_x, max_y, max_z],
                    "axial_slices_span_z": [min_z, max_z],
                    "coronal_slices_span_y": [min_y, max_y],
                    "sagittal_slices_span_x": [min_x, max_x],
                    "mean_intensity": mean_val
                })

        metrics = self.compute_metrics(old_mask, old_mask, context)
        metrics["total_islands"] = int(num_features)
        metrics["debris_count"] = debris_count
        metrics["islands_analyzed"] = islands_list
        metrics["connectivity_status"] = "SINGLE_CONNECTED_OBJECT" if num_features == 1 else "FRAGMENTED_MULTIPLE_ISLANDS"

        msg = (
            f"连通性扫描完成: 检测到 {num_features} 个独立连通域。"
            f"最大主体积 {islands_list[0]['volume_cm3'] if islands_list else 0} cm³ (占比 {islands_list[0]['volume_ratio_pct'] if islands_list else 0}%)。"
        )
        if num_features > 1:
            msg += f" 检测到 {num_features - 1} 个离散碎屑/粘连分支，建议根据质心坐标进行擦除或调用 filter_connected_components 过滤。"

        return ToolResult(
            success=True,
            new_mask=old_mask,
            observation_metrics=metrics,
            action_description="主动连通性扫描与拓扑诊断",
            message=msg
        )


class ConnectedComponentFilterTool(BaseMedicalTool):
    """三维连通域分析与多维度条件过滤算子"""
    @property
    def name(self) -> str:
        return "filter_connected_components"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 连通域精准过滤】支持多维度去噪与孤岛保留模式:\n"
            "1. 按数量保留: `keep_top_k=1` (仅保留最大单一主体);\n"
            "2. 按体积过滤: `min_volume_mm3=500.0` (清除小于该体积的碎片);\n"
            "3. 按点包含筛选: `keep_point_3d=[X,Y,Z]` 或 `keep_point_2d` (仅保留包含该目标点的连通分支);\n"
            "4. 按编号移除: `remove_island_ids=[2, 3]` (精准移除指定碎屑孤岛)。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min_volume_mm3": {
                    "type": "number",
                    "description": "过滤的最小物理体积阈值 (mm³)。小于该值的孤立碎片将被清除。"
                },
                "keep_top_k": {
                    "type": "integer",
                    "description": "仅保留体积最大的前 K 个主要连通域 (例如 1 表示只保留最大单一主体)。"
                },
                "keep_point_3d": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "三维坐标点 [X, Y, Z]: 仅保留包覆该点击点的特定连通域分支。"
                },
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "可选: 切片平面 ('axial', 'coronal', 'sagittal')，配合 slice_index 与 point_2d 使用。"
                },
                "slice_index": {
                    "type": "integer",
                    "description": "可选: 切片绝对索引。"
                },
                "point_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "可选: 切片二维坐标点 [c1, c2]，仅保留包含该切片点击点的连通域。"
                },
                "remove_island_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "要精准移除的连通域编号列表 (按体积排序从 1 开始，如 [2, 3])。"
                }
            }
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask.copy()
        dim_x, dim_y, dim_z = context.shape
        if np.count_nonzero(old_mask) == 0:
            return ToolResult(
                success=False,
                new_mask=old_mask,
                observation_metrics=self.compute_metrics(old_mask, old_mask, context),
                message="当前工作区中没有已激活的 Mask，无需去噪。"
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
        sorted_indices = np.argsort(sizes)[::-1]
        new_mask = np.zeros_like(old_mask)

        # 模式 1: 按点包含筛选 (keep_point_3d 或 2D 切片点)
        target_point = None
        if "keep_point_3d" in kwargs and len(kwargs["keep_point_3d"]) == 3:
            target_point = [int(kwargs["keep_point_3d"][0]), int(kwargs["keep_point_3d"][1]), int(kwargs["keep_point_3d"][2])]
        else:
            plane = kwargs.get("plane", "").lower()
            s_idx = kwargs.get("slice_index", None)
            p2d = kwargs.get("point_2d", None)
            if plane and s_idx is not None and p2d and len(p2d) >= 2:
                s_idx = int(s_idx)
                c1, c2 = int(round(p2d[0])), int(round(p2d[1]))
                if plane == "sagittal":
                    target_point = [np.clip(s_idx, 0, dim_x - 1), np.clip(c1, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)]
                elif plane == "coronal":
                    target_point = [np.clip(c1, 0, dim_x - 1), np.clip(s_idx, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)]
                elif plane == "axial":
                    target_point = [np.clip(c1, 0, dim_x - 1), np.clip(c2, 0, dim_y - 1), np.clip(s_idx, 0, dim_z - 1)]

        if target_point is not None:
            tx, ty, tz = target_point
            target_label = labeled_array[tx, ty, tz]
            if target_label > 0:
                new_mask[labeled_array == target_label] = 1
                desc = f"仅保留包覆目标点 [{tx}, {ty}, {tz}] 的特定连通分支 (Label {target_label})"
            else:
                return ToolResult(success=False, new_mask=old_mask, message=f"目标点 [{tx}, {ty}, {tz}] 处未落在任何掩码连通域上")

        # 模式 2: 按指定编号移除 (remove_island_ids)
        elif "remove_island_ids" in kwargs and kwargs["remove_island_ids"]:
            remove_ranks = set(kwargs["remove_island_ids"])
            new_mask = old_mask.copy()
            for r in remove_ranks:
                if 1 <= r <= len(sorted_indices):
                    idx = sorted_indices[r - 1]
                    lab = idx + 1
                    new_mask[labeled_array == lab] = 0
            desc = f"精准移除指定连通域编号: {list(remove_ranks)}"

        # 模式 3: 保留 Top K
        elif kwargs.get("keep_top_k") is not None and int(kwargs.get("keep_top_k")) > 0:
            k = int(kwargs.get("keep_top_k"))
            top_labels = [sorted_indices[i] + 1 for i in range(min(k, len(sorted_indices)))]
            for lab in top_labels:
                new_mask[labeled_array == lab] = 1
            desc = f"保留体积最大的前 {k} 个主要连通域"

        # 模式 4: 按最小体积阈值过滤
        else:
            min_volume_mm3 = float(kwargs.get("min_volume_mm3", 50.0))
            min_voxels = max(1, int(round(min_volume_mm3 / context.voxel_volume_mm3)))
            valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
            for lab in valid_labels:
                new_mask[labeled_array == lab] = 1
            desc = f"过滤小于 {min_volume_mm3} mm³ 的微小碎屑 (共过滤 {num_features - len(valid_labels)} 个噪点)"

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"连通域精准过滤 ({desc})",
            message=f"已完成连通域过滤 ({desc})。清理体积 {abs(metrics['volume_change_mm3'])} mm³。"
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
