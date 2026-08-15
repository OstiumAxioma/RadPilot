import numpy as np
from scipy import ndimage
from typing import Dict, Any, List, Optional
from .base_tool import BaseMedicalTool, ImageContext, ToolResult

class ThresholdRangeTool(BaseMedicalTool):
    """3D 信号强度/HU 阈值窗范围卡取算子"""
    @property
    def name(self) -> str:
        return "threshold_range"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 阈值卡取】在 3D 图像全图或指定空间 ROI 内，提取满足 [min_intensity, max_intensity] 信号区间的体素。"
            "常用于由粗到细的第一步：先卡出目标组织或病灶的大致灰度范围。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min_intensity": {
                    "type": "number",
                    "description": "信号强度下限 (如 2500.0)"
                },
                "max_intensity": {
                    "type": "number",
                    "description": "信号强度上限 (如 6500.0)"
                },
                "bbox_3d": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "可选的局部空间范围 [x_min, y_min, z_min, x_max, y_max, z_max]，不传则默认全图计算"
                },
                "mode": {
                    "type": "string",
                    "enum": ["replace", "add", "intersect"],
                    "description": "操作模式: 'replace'(作为全新掩码), 'add'(合并至已有掩码), 'intersect'(仅在已有掩码内做交集过滤)。默认 'replace'。"
                }
            },
            "required": ["min_intensity", "max_intensity"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        img = context.image_data
        dim_x, dim_y, dim_z = context.shape

        min_val = float(kwargs.get("min_intensity", 0.0))
        max_val = float(kwargs.get("max_intensity", 99999.0))
        raw_bbox = kwargs.get("bbox_3d", None)
        mode = kwargs.get("mode", "replace")

        # 计算全局阈值图
        thresh_pass = np.logical_and(img >= min_val, img <= max_val)

        if raw_bbox and len(raw_bbox) == 6:
            bx0 = max(0, min(raw_bbox[0], raw_bbox[3]))
            bx1 = min(dim_x, max(raw_bbox[0], raw_bbox[3]))
            by0 = max(0, min(raw_bbox[1], raw_bbox[4]))
            by1 = min(dim_y, max(raw_bbox[1], raw_bbox[4]))
            bz0 = max(0, min(raw_bbox[2], raw_bbox[5]))
            bz1 = min(dim_z, max(raw_bbox[2], raw_bbox[5]))

            roi_mask = np.zeros_like(thresh_pass, dtype=bool)
            roi_mask[bx0:bx1, by0:by1, bz0:bz1] = True
            thresh_pass = np.logical_and(thresh_pass, roi_mask)

        extracted = thresh_pass.astype(np.uint8)

        if mode == "add":
            new_mask = np.logical_or(old_mask, extracted).astype(np.uint8)
        elif mode == "intersect":
            new_mask = np.logical_and(old_mask, extracted).astype(np.uint8)
        else:
            new_mask = extracted

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["threshold_range"] = [min_val, max_val]

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"阈值卡取 [{min_val}, {max_val}], 模式={mode}"
        )


class PaintBrush3DTool(BaseMedicalTool):
    """三维空间球形画笔填色涂抹算子"""
    @property
    def name(self) -> str:
        return "paint_brush_3d"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 3D画笔】在指定三维空间坐标中心 [x, y, z] 上，以指定物理半径 (radius_mm) 进行 3D 球形画笔填色涂抹。"
            "常用于手动修补漏勾画的解剖边缘或微小结构。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "center": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "画笔球心三维坐标 [x, y, z]"
                },
                "radius_mm": {
                    "type": "number",
                    "description": "画笔物理半径 (毫米 mm，如 4.0, 6.5)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["add", "replace"],
                    "description": "绘制模式: 'add'(向已有掩码中追加), 'replace'(仅保留本画笔笔触)。默认 'add'。"
                }
            },
            "required": ["center", "radius_mm"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        dim_x, dim_y, dim_z = context.shape
        sp_x, sp_y, sp_z = context.spacing

        center = kwargs.get("center", [dim_x//2, dim_y//2, dim_z//2])
        radius_mm = float(kwargs.get("radius_mm", 5.0))
        mode = kwargs.get("mode", "add")

        cx, cy, cz = int(center[0]), int(center[1]), int(center[2])
        rx = max(1, int(round(radius_mm / sp_x)))
        ry = max(1, int(round(radius_mm / sp_y)))
        rz = max(1, int(round(radius_mm / sp_z)))

        x0, x1 = max(0, cx - rx), min(dim_x, cx + rx + 1)
        y0, y1 = max(0, cy - ry), min(dim_y, cy + ry + 1)
        z0, z1 = max(0, cz - rz), min(dim_z, cz + rz + 1)

        x_grid, y_grid, z_grid = np.ogrid[x0-cx:x1-cx, y0-cy:y1-cy, z0-cz:z1-cz]
        sphere_patch = ((x_grid * sp_x)**2 + (y_grid * sp_y)**2 + (z_grid * sp_z)**2) <= (radius_mm**2)

        brush_mask = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
        brush_mask[x0:x1, y0:y1, z0:z1] = sphere_patch.astype(np.uint8)

        if mode == "replace":
            new_mask = brush_mask
        else:
            new_mask = np.logical_or(old_mask, brush_mask).astype(np.uint8)

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["brush_center"] = [cx, cy, cz]
        metrics["brush_radius_mm"] = radius_mm

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"3D画笔涂抹 (中心 [{cx}, {cy}, {cz}], 半径 {radius_mm}mm)"
        )


class EraseBrush3DTool(BaseMedicalTool):
    """三维空间球形橡皮擦擦除算子"""
    @property
    def name(self) -> str:
        return "erase_brush_3d"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 3D橡皮擦】在指定三维空间坐标中心 [x, y, z] 上，以指定物理半径 (radius_mm) 进行 3D 球形擦除。"
            "常用于精准剔除粘连组织、颅骨伪影或误包含的非目标区域。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "center": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "橡皮擦球心三维坐标 [x, y, z]"
                },
                "radius_mm": {
                    "type": "number",
                    "description": "橡皮擦物理半径 (毫米 mm，如 5.0, 8.0)"
                }
            },
            "required": ["center", "radius_mm"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        dim_x, dim_y, dim_z = context.shape
        sp_x, sp_y, sp_z = context.spacing

        center = kwargs.get("center", [dim_x//2, dim_y//2, dim_z//2])
        radius_mm = float(kwargs.get("radius_mm", 5.0))

        cx, cy, cz = int(center[0]), int(center[1]), int(center[2])
        rx = max(1, int(round(radius_mm / sp_x)))
        ry = max(1, int(round(radius_mm / sp_y)))
        rz = max(1, int(round(radius_mm / sp_z)))

        x0, x1 = max(0, cx - rx), min(dim_x, cx + rx + 1)
        y0, y1 = max(0, cy - ry), min(dim_y, cy + ry + 1)
        z0, z1 = max(0, cz - rz), min(dim_z, cz + rz + 1)

        x_grid, y_grid, z_grid = np.ogrid[x0-cx:x1-cx, y0-cy:y1-cy, z0-cz:z1-cz]
        sphere_patch = ((x_grid * sp_x)**2 + (y_grid * sp_y)**2 + (z_grid * sp_z)**2) <= (radius_mm**2)

        erase_mask = np.ones((dim_x, dim_y, dim_z), dtype=np.uint8)
        erase_mask[x0:x1, y0:y1, z0:z1] = np.logical_not(sphere_patch).astype(np.uint8)

        new_mask = np.logical_and(old_mask, erase_mask).astype(np.uint8)

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["erase_center"] = [cx, cy, cz]
        metrics["erase_radius_mm"] = radius_mm

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"3D橡皮擦擦除 (中心 [{cx}, {cy}, {cz}], 半径 {radius_mm}mm)"
        )


class ScissorsCutTool(BaseMedicalTool):
    """解剖空间剪刀剪除算子"""
    @property
    def name(self) -> str:
        return "scissors_cut"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 空间切刀】沿着指定的解剖平面 (Axial Z轴, Coronal Y轴, Sagittal X轴) 将当前掩码一刀切断，并移除指定一侧的组织。"
            "例如: 沿冠状位 Y=115 切断，移除前方 (Anterior) 的脑干组织，仅保留后方 (Posterior) 的小脑。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "剪切参考平面: 'axial'(水平Z轴横切), 'coronal'(冠状Y轴前后切), 'sagittal'(矢状X轴左右切)"
                },
                "cut_index": {
                    "type": "integer",
                    "description": "切刀所在的切片坐标索引 (如 Y=115, Z=45)"
                },
                "remove_side": {
                    "type": "string",
                    "enum": ["greater_than", "less_than"],
                    "description": "需要移除的一侧: 'greater_than'(移除坐标大于cut_index的区域), 'less_than'(移除坐标小于cut_index的区域)"
                }
            },
            "required": ["plane", "cut_index", "remove_side"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        dim_x, dim_y, dim_z = context.shape

        plane = kwargs.get("plane", "coronal")
        cut_idx = int(kwargs.get("cut_index", 0))
        remove_side = kwargs.get("remove_side", "greater_than")

        new_mask = old_mask.copy()

        if plane == "axial":
            idx = max(0, min(dim_z - 1, cut_idx))
            if remove_side == "greater_than":
                new_mask[:, :, idx:] = 0
            else:
                new_mask[:, :, :idx] = 0
            desc = f"沿轴位 Z={idx} 切割并移除 {remove_side}"

        elif plane == "coronal":
            idx = max(0, min(dim_y - 1, cut_idx))
            if remove_side == "greater_than":
                new_mask[:, idx:, :] = 0
            else:
                new_mask[:, :idx, :] = 0
            desc = f"沿冠状位 Y={idx} 切割并移除 {remove_side}"

        else:
            idx = max(0, min(dim_x - 1, cut_idx))
            if remove_side == "greater_than":
                new_mask[idx:, :, :] = 0
            else:
                new_mask[:idx, :, :] = 0
            desc = f"沿矢状位 X={idx} 切割并移除 {remove_side}"

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"空间切刀: {desc}"
        )


class RegionGrowthTool(BaseMedicalTool):
    """3D 自适应梯度阻尼区域生长算子"""
    @property
    def name(self) -> str:
        return "region_growth"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 3D区域生长】从指定的 3D 种子坐标 [x, y, z] 开始，以指定的强度容差 (tolerance) 并在 3D 物理梯度阻尼下自然漫延。"
            "遇到脑脊液、小脑幕或骨骼分界面自动阻断，生成光滑贴合的生物解剖轮廓。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "seed_point": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "生长起始 3D 种子点坐标 [x, y, z]"
                },
                "intensity_tolerance": {
                    "type": "number",
                    "description": "允许的信号灰度容差范围 (如 600.0, 1200.0)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["replace", "add", "subtract"],
                    "description": "合并模式: 'replace'(独立生成), 'add'(增量并入), 'subtract'(扣除)。默认 'replace'。"
                }
            },
            "required": ["seed_point"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        img = context.image_data.astype(np.float32)
        dim_x, dim_y, dim_z = context.shape

        seed = kwargs.get("seed_point", [dim_x//2, dim_y//2, dim_z//2])
        tolerance = float(kwargs.get("intensity_tolerance", 800.0))
        mode = kwargs.get("mode", "replace")

        sx = int(np.clip(seed[0], 0, dim_x - 1))
        sy = int(np.clip(seed[1], 0, dim_y - 1))
        sz = int(np.clip(seed[2], 0, dim_z - 1))

        smoothed = ndimage.gaussian_filter(img, sigma=1.0)
        seed_val = smoothed[sx, sy, sz]

        # 3D 梯度场阻尼
        gx = ndimage.sobel(smoothed, axis=0)
        gy = ndimage.sobel(smoothed, axis=1)
        gz = ndimage.sobel(smoothed, axis=2)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        grad_thresh = np.percentile(grad_mag, 90)

        # 强度通行矩阵
        pass_matrix = np.logical_and(
            np.abs(smoothed - seed_val) <= tolerance,
            grad_mag < grad_thresh * 1.5
        ).astype(np.uint8)
        pass_matrix[sx, sy, sz] = 1

        # 连通域提取
        labeled, num_features = ndimage.label(pass_matrix)
        seed_label = labeled[sx, sy, sz]
        if seed_label > 0:
            grown = (labeled == seed_label).astype(np.uint8)
        else:
            grown = pass_matrix

        # 孔洞充填与微闭运算
        grown = ndimage.binary_fill_holes(grown).astype(np.uint8)
        struct = ndimage.generate_binary_structure(3, 1)
        grown = ndimage.binary_closing(grown, structure=struct, iterations=1).astype(np.uint8)

        if mode == "add":
            new_mask = np.logical_or(old_mask, grown).astype(np.uint8)
        elif mode == "subtract":
            new_mask = np.logical_and(old_mask, np.logical_not(grown)).astype(np.uint8)
        else:
            new_mask = grown

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["growth_seed"] = [sx, sy, sz]
        metrics["tolerance"] = tolerance

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"3D区域生长 (种子 [{sx}, {sy}, {sz}], 容差 {tolerance})"
        )


class FillBetweenSlicesTool(BaseMedicalTool):
    """关键断层切片间形态学插值算子 (Fill Between Slices)"""
    @property
    def name(self) -> str:
        return "fill_between_slices"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 切片间插值】在两个指定断层 (如 slice_start 与 slice_end) 之间，"
            "根据两端的掩码轮廓进行三维形态学凸包与测地线连续形态学插值，自动填补中间所有空缺层。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "axis": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "插值主轴向: 'axial'(沿着Z轴切片间插值), 'coronal'(Y轴), 'sagittal'(X轴)。默认 'axial'。"
                },
                "slice_start": {
                    "type": "integer",
                    "description": "起始切片索引 (如 Z=20)"
                },
                "slice_end": {
                    "type": "integer",
                    "description": "结束切片索引 (如 Z=45)"
                }
            },
            "required": ["slice_start", "slice_end"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        dim_x, dim_y, dim_z = context.shape
        axis = kwargs.get("axis", "axial")

        s0 = int(kwargs.get("slice_start", 0))
        s1 = int(kwargs.get("slice_end", 10))
        start_idx = min(s0, s1)
        end_idx = max(s0, s1)

        new_mask = old_mask.copy()

        if axis == "axial":
            start_idx = max(0, min(dim_z - 1, start_idx))
            end_idx = max(0, min(dim_z - 1, end_idx))
            if end_idx - start_idx >= 2:
                m_start = new_mask[:, :, start_idx]
                m_end = new_mask[:, :, end_idx]
                # 若两端均有标记，则进行形态学线性插值
                for z in range(start_idx + 1, end_idx):
                    alpha = (z - start_idx) / float(end_idx - start_idx)
                    interp_slice = np.logical_or(
                        (m_start > 0) if (1.0 - alpha) > 0.5 else False,
                        (m_end > 0) if alpha > 0.5 else False
                    ).astype(np.uint8)
                    # 结合两者逻辑与和形态学闭运算
                    comb = np.logical_and(m_start > 0, m_end > 0)
                    interp_slice = np.logical_or(interp_slice, comb).astype(np.uint8)
                    interp_slice = ndimage.binary_fill_holes(interp_slice).astype(np.uint8)
                    new_mask[:, :, z] = interp_slice

        elif axis == "coronal":
            start_idx = max(0, min(dim_y - 1, start_idx))
            end_idx = max(0, min(dim_y - 1, end_idx))
            if end_idx - start_idx >= 2:
                m_start = new_mask[:, start_idx, :]
                m_end = new_mask[:, end_idx, :]
                for y in range(start_idx + 1, end_idx):
                    alpha = (y - start_idx) / float(end_idx - start_idx)
                    interp_slice = np.logical_or(
                        (m_start > 0) if (1.0 - alpha) > 0.5 else False,
                        (m_end > 0) if alpha > 0.5 else False
                    ).astype(np.uint8)
                    interp_slice = ndimage.binary_fill_holes(interp_slice).astype(np.uint8)
                    new_mask[:, y, :] = interp_slice

        else:
            start_idx = max(0, min(dim_x - 1, start_idx))
            end_idx = max(0, min(dim_x - 1, end_idx))
            if end_idx - start_idx >= 2:
                m_start = new_mask[start_idx, :, :]
                m_end = new_mask[end_idx, :, :]
                for x in range(start_idx + 1, end_idx):
                    alpha = (x - start_idx) / float(end_idx - start_idx)
                    interp_slice = np.logical_or(
                        (m_start > 0) if (1.0 - alpha) > 0.5 else False,
                        (m_end > 0) if alpha > 0.5 else False
                    ).astype(np.uint8)
                    interp_slice = ndimage.binary_fill_holes(interp_slice).astype(np.uint8)
                    new_mask[x, :, :] = interp_slice

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"切片间形态学插值 ({axis}: {start_idx} -> {end_idx})"
        )


class IslandAndSmoothTool(BaseMedicalTool):
    """孤岛碎屑过滤与生物曲率平滑填孔算子"""
    @property
    def name(self) -> str:
        return "island_and_smooth"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 孤岛提纯与平滑】过滤掉小于指定物理体积 (min_volume_mm3) 的所有离散伪影孤岛，"
            "并执行 3D 孔洞填充与曲率平滑闭运算。常用于精细微调的最后一步收尾工作。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min_volume_mm3": {
                    "type": "number",
                    "description": "过滤碎屑的最小物理体积阈值 (mm³，默认 50.0)"
                },
                "keep_top_k": {
                    "type": "integer",
                    "description": "仅保留最大的前 K 个连通域分支 (如 1 或 2，不传则按体积过滤)"
                },
                "fill_holes": {
                    "type": "boolean",
                    "description": "是否充填内部微孔。默认 True。"
                }
            },
            "required": []
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        min_vol = float(kwargs.get("min_volume_mm3", 50.0))
        keep_top_k = kwargs.get("keep_top_k", None)
        fill_holes = kwargs.get("fill_holes", True)
        vox_vol = context.voxel_volume_mm3

        labeled, num_features = ndimage.label(old_mask)
        if num_features == 0:
            return ToolResult(success=True, new_mask=old_mask, observation_metrics=self.compute_metrics(old_mask, old_mask, context), message="当前掩码为空，无需提纯")

        sizes = ndimage.sum(old_mask, labeled, range(1, num_features + 1))

        if keep_top_k and keep_top_k > 0:
            k = min(int(keep_top_k), num_features)
            sorted_indices = np.argsort(sizes)[::-1]
            top_labels = set(sorted_indices[:k] + 1)
            filtered = np.isin(labeled, list(top_labels)).astype(np.uint8)
        else:
            min_voxels = max(1, int(round(min_vol / vox_vol)))
            keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
            filtered = np.isin(labeled, keep_labels).astype(np.uint8)

        if fill_holes:
            filtered = ndimage.binary_fill_holes(filtered).astype(np.uint8)

        struct = ndimage.generate_binary_structure(3, 1)
        filtered = ndimage.binary_closing(filtered, structure=struct, iterations=1).astype(np.uint8)

        metrics = self.compute_metrics(old_mask, filtered, context)
        return ToolResult(
            success=True,
            new_mask=filtered,
            observation_metrics=metrics,
            action_description=f"孤岛提纯与平滑 (过滤<{min_vol}mm³ 碎屑, 填孔={fill_holes})"
        )
