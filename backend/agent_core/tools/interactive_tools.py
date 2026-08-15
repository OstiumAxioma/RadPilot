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
    """三维/切片双模态球形画笔填色涂抹算子"""
    @property
    def name(self) -> str:
        return "paint_brush_3d"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 局部精准画笔 (Paint Brush)】在三维空间或指定切片断面上进行局部球形/圆盘填色涂抹。"
            "💡【微调精雕首选】: 当掩码仅有局部边缘微小漏包、欠分割或孔洞时，严禁重拉阈值！直接在对应断面上用画笔点补几处即可！"
            "支持直接传入切片二维坐标 (plane + slice_index + point_2d)，或直接传入三维空间坐标 (center)。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "可选: 当前涂抹的切片视角 ('axial', 'coronal', 'sagittal')。"
                },
                "slice_index": {
                    "type": "integer",
                    "description": "可选: 切片绝对索引 (如轴位 Z=35, 矢状面 X=91)。"
                },
                "point_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "可选: 切片二维坐标点 [c1, c2]。矢状位传入 [Y, Z]，轴位传入 [X, Y]，冠状位传入 [X, Z]。"
                },
                "points_2d": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "可选: 批量切片二维涂抹点序列 [[c1, c2], ...]。"
                },
                "center": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "可选: 画笔球心三维坐标 [x, y, z]。"
                },
                "centers": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "可选: 批量三维涂抹点序列 [[x, y, z], ...]。"
                },
                "radius_mm": {
                    "type": "number",
                    "description": "画笔物理半径 (毫米 mm，如 3.0, 5.0, 8.0)。默认 5.0。"
                },
                "mode": {
                    "type": "string",
                    "enum": ["add", "replace"],
                    "description": "绘制模式: 'add'(追加填色), 'replace'(替换)。默认 'add'。"
                }
            }
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask.copy()
        dim_x, dim_y, dim_z = context.shape
        sp_x, sp_y, sp_z = context.spacing
        radius_mm = float(kwargs.get("radius_mm", 5.0))
        mode = kwargs.get("mode", "add")

        # 解析多模态涂抹中心点序列
        center_points = []
        if "center" in kwargs and len(kwargs["center"]) == 3:
            center_points.append([int(kwargs["center"][0]), int(kwargs["center"][1]), int(kwargs["center"][2])])
        if "centers" in kwargs:
            for c in kwargs["centers"]:
                if len(c) == 3:
                    center_points.append([int(c[0]), int(c[1]), int(c[2])])

        # 2D 切片投影点解析
        plane = kwargs.get("plane", "").lower()
        s_idx = kwargs.get("slice_index", None)
        p2d = kwargs.get("point_2d", None)
        p2ds = kwargs.get("points_2d", [])
        if p2d:
            p2ds = [p2d] + list(p2ds)

        if plane and s_idx is not None and p2ds:
            s_idx = int(s_idx)
            for pt in p2ds:
                if len(pt) >= 2:
                    c1, c2 = int(round(pt[0])), int(round(pt[1]))
                    if plane == "sagittal":
                        center_points.append([np.clip(s_idx, 0, dim_x - 1), np.clip(c1, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)])
                    elif plane == "coronal":
                        center_points.append([np.clip(c1, 0, dim_x - 1), np.clip(s_idx, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)])
                    elif plane == "axial":
                        center_points.append([np.clip(c1, 0, dim_x - 1), np.clip(c2, 0, dim_y - 1), np.clip(s_idx, 0, dim_z - 1)])

        if not center_points:
            center_points = [[dim_x // 2, dim_y // 2, dim_z // 2]]

        rx = max(1, int(round(radius_mm / sp_x)))
        ry = max(1, int(round(radius_mm / sp_y)))
        rz = max(1, int(round(radius_mm / sp_z)))

        brush_total = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
        for cx, cy, cz in center_points:
            x0, x1 = max(0, cx - rx), min(dim_x, cx + rx + 1)
            y0, y1 = max(0, cy - ry), min(dim_y, cy + ry + 1)
            z0, z1 = max(0, cz - rz), min(dim_z, cz + rz + 1)

            x_grid, y_grid, z_grid = np.ogrid[x0-cx:x1-cx, y0-cy:y1-cy, z0-cz:z1-cz]
            sphere_patch = ((x_grid * sp_x)**2 + (y_grid * sp_y)**2 + (z_grid * sp_z)**2) <= (radius_mm**2)
            brush_total[x0:x1, y0:y1, z0:z1] = np.logical_or(brush_total[x0:x1, y0:y1, z0:z1], sphere_patch).astype(np.uint8)

        if mode == "replace":
            new_mask = brush_total
        else:
            new_mask = np.logical_or(old_mask, brush_total).astype(np.uint8)

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["brush_centers"] = center_points
        metrics["brush_radius_mm"] = radius_mm

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"局部画笔填色涂抹 (点位数={len(center_points)}, 半径={radius_mm}mm)"
        )


class EraseBrush3DTool(BaseMedicalTool):
    """三维/切片双模态球形橡皮擦擦除算子"""
    @property
    def name(self) -> str:
        return "erase_brush_3d"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 局部精准橡皮擦 (Erase Brush)】在三维空间或指定切片断面上进行局部球形擦除。"
            "💡【微调精雕首选】: 当掩码仅有微小的局部突刺、边缘毛刺、轻微粘连或邻近小结构溢出时，严禁使用大剪刀粗暴全切！直接用橡皮擦在突刺位置擦除即可！"
            "支持直接传入切片二维坐标 (plane + slice_index + point_2d)，或直接传入三维空间坐标 (center)。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "可选: 当前擦除的切片视角 ('axial', 'coronal', 'sagittal')。"
                },
                "slice_index": {
                    "type": "integer",
                    "description": "可选: 切片绝对索引 (如轴位 Z=35, 矢状面 X=91)。"
                },
                "point_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "可选: 切片二维坐标点 [c1, c2]。矢状位传入 [Y, Z]，轴位传入 [X, Y]，冠状位传入 [X, Z]。"
                },
                "points_2d": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "可选: 批量切片二维擦除点序列 [[c1, c2], ...]。"
                },
                "center": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "可选: 橡皮擦球心三维坐标 [x, y, z]。"
                },
                "centers": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "可选: 批量三维擦除点序列 [[x, y, z], ...]。"
                },
                "radius_mm": {
                    "type": "number",
                    "description": "橡皮擦物理半径 (毫米 mm，如 3.0, 5.0, 8.0)。默认 5.0。"
                }
            }
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask.copy()
        dim_x, dim_y, dim_z = context.shape
        sp_x, sp_y, sp_z = context.spacing
        radius_mm = float(kwargs.get("radius_mm", 5.0))

        center_points = []
        if "center" in kwargs and len(kwargs["center"]) == 3:
            center_points.append([int(kwargs["center"][0]), int(kwargs["center"][1]), int(kwargs["center"][2])])
        if "centers" in kwargs:
            for c in kwargs["centers"]:
                if len(c) == 3:
                    center_points.append([int(c[0]), int(c[1]), int(c[2])])

        plane = kwargs.get("plane", "").lower()
        s_idx = kwargs.get("slice_index", None)
        p2d = kwargs.get("point_2d", None)
        p2ds = kwargs.get("points_2d", [])
        if p2d:
            p2ds = [p2d] + list(p2ds)

        if plane and s_idx is not None and p2ds:
            s_idx = int(s_idx)
            for pt in p2ds:
                if len(pt) >= 2:
                    c1, c2 = int(round(pt[0])), int(round(pt[1]))
                    if plane == "sagittal":
                        center_points.append([np.clip(s_idx, 0, dim_x - 1), np.clip(c1, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)])
                    elif plane == "coronal":
                        center_points.append([np.clip(c1, 0, dim_x - 1), np.clip(s_idx, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)])
                    elif plane == "axial":
                        center_points.append([np.clip(c1, 0, dim_x - 1), np.clip(c2, 0, dim_y - 1), np.clip(s_idx, 0, dim_z - 1)])

        if not center_points:
            center_points = [[dim_x // 2, dim_y // 2, dim_z // 2]]

        rx = max(1, int(round(radius_mm / sp_x)))
        ry = max(1, int(round(radius_mm / sp_y)))
        rz = max(1, int(round(radius_mm / sp_z)))

        erase_total = np.ones((dim_x, dim_y, dim_z), dtype=np.uint8)
        for cx, cy, cz in center_points:
            x0, x1 = max(0, cx - rx), min(dim_x, cx + rx + 1)
            y0, y1 = max(0, cy - ry), min(dim_y, cy + ry + 1)
            z0, z1 = max(0, cz - rz), min(dim_z, cz + rz + 1)

            x_grid, y_grid, z_grid = np.ogrid[x0-cx:x1-cx, y0-cy:y1-cy, z0-cz:z1-cz]
            sphere_patch = ((x_grid * sp_x)**2 + (y_grid * sp_y)**2 + (z_grid * sp_z)**2) <= (radius_mm**2)
            erase_total[x0:x1, y0:y1, z0:z1] = np.logical_and(erase_total[x0:x1, y0:y1, z0:z1], np.logical_not(sphere_patch)).astype(np.uint8)

        new_mask = np.logical_and(old_mask, erase_total).astype(np.uint8)

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["erase_centers"] = center_points
        metrics["erase_radius_mm"] = radius_mm

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"局部橡皮擦擦除 (点位数={len(center_points)}, 半径={radius_mm}mm)"
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


def _rasterize_polygon_2d(shape_2d: tuple, points: List[List[float]]) -> np.ndarray:
    """
    通用高精度 2D 多边形光栅化函数 (支持绝对坐标与 0~1 归一化浮点坐标)
    """
    dim_h, dim_w = shape_2d
    mask = np.zeros((dim_h, dim_w), dtype=np.uint8)
    if not points or len(points) < 3:
        return mask

    pts = np.array(points, dtype=np.float32)
    # 若为 0~1 归一化比例坐标，转换为像素坐标
    if np.max(pts) <= 1.0 and np.min(pts) >= 0.0:
        pts[:, 0] = pts[:, 0] * (dim_w - 1)
        pts[:, 1] = pts[:, 1] * (dim_h - 1)

    try:
        from matplotlib.path import Path
        y_grid, x_grid = np.mgrid[:dim_h, :dim_w]
        grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        poly_path = Path(pts)
        inside = poly_path.contains_points(grid_points)
        mask = inside.reshape((dim_h, dim_w)).astype(np.uint8)
    except Exception:
        # 回退实现: 简单扫描线多边形填充
        import cv2
        int_pts = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [int_pts], 1)

    return mask


class DrawPolygonContourTool(BaseMedicalTool):
    """连续多边形解剖轮廓绘制算子 (Draw Freehand / Polygon Contour)"""
    @property
    def name(self) -> str:
        return "draw_polygon_contour"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 连续多边形绘制 (Draw Contour)】在指定正交切片断面 (矢状位/冠状位/轴位) 上绘制连续闭合多边形解剖轮廓。"
            "大模型可直接像放射科专家一样传入有机曲线控制点序列 [[c1, c2], [c3, c4], ...]，彻底告别死板的长方体 Bounding Box！"
            "系统会自动将多边形高精度栅格化并合并至 3D 掩码中。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "绘制切片平面: 'sagittal'(矢状位, 如正中矢状面 X=91), 'coronal'(冠状位), 'axial'(轴位)。"
                },
                "slice_index": {
                    "type": "integer",
                    "description": "断层切片绝对索引 (如矢状面 X=91, 轴位 Z=45 等)"
                },
                "points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "多边形闭合控制点序列。矢状位传入 [[Y, Z], ...], 冠状位传入 [[X, Z], ...], 轴位传入 [[X, Y], ...]。"
                },
                "thickness_voxels": {
                    "type": "integer",
                    "description": "沿垂直切片方向扩展的厚度体素数 (默认 1，可设置为 2~5 层)。"
                },
                "mode": {
                    "type": "string",
                    "enum": ["add", "replace", "intersect"],
                    "description": "操作模式: 'add'(合并), 'replace'(替换), 'intersect'(求交)。默认 'add'。"
                }
            },
            "required": ["plane", "slice_index", "points"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask.copy()
        dim_x, dim_y, dim_z = context.shape
        plane = kwargs.get("plane", "sagittal").lower()
        slice_idx = int(kwargs.get("slice_index", 0))
        points = kwargs.get("points", [])
        thickness = max(1, int(kwargs.get("thickness_voxels", 1)))
        mode = kwargs.get("mode", "add")

        if len(points) < 3:
            return ToolResult(success=False, new_mask=old_mask, message="多边形控制点不足 3 个，无法构成闭合轮廓")

        half_t = thickness // 2
        new_mask = old_mask.copy()

        if plane == "sagittal":
            slice_idx = np.clip(slice_idx, 0, dim_x - 1)
            poly_2d = _rasterize_polygon_2d((dim_z, dim_y), points)
            # 转为 (dim_y, dim_z) 矩阵
            poly_yz = poly_2d.T
            x_start = max(0, slice_idx - half_t)
            x_end = min(dim_x, slice_idx + half_t + (thickness % 2))
            for x in range(x_start, x_end):
                if mode == "add":
                    new_mask[x, :, :] = np.logical_or(new_mask[x, :, :], poly_yz).astype(np.uint8)
                elif mode == "replace":
                    new_mask[x, :, :] = poly_yz
                elif mode == "intersect":
                    new_mask[x, :, :] = np.logical_and(new_mask[x, :, :], poly_yz).astype(np.uint8)

        elif plane == "coronal":
            slice_idx = np.clip(slice_idx, 0, dim_y - 1)
            poly_2d = _rasterize_polygon_2d((dim_z, dim_x), points)
            poly_xz = poly_2d.T
            y_start = max(0, slice_idx - half_t)
            y_end = min(dim_y, slice_idx + half_t + (thickness % 2))
            for y in range(y_start, y_end):
                if mode == "add":
                    new_mask[:, y, :] = np.logical_or(new_mask[:, y, :], poly_xz).astype(np.uint8)
                elif mode == "replace":
                    new_mask[:, y, :] = poly_xz
                elif mode == "intersect":
                    new_mask[:, y, :] = np.logical_and(new_mask[:, y, :], poly_xz).astype(np.uint8)

        elif plane == "axial":
            slice_idx = np.clip(slice_idx, 0, dim_z - 1)
            poly_2d = _rasterize_polygon_2d((dim_y, dim_x), points)
            poly_xy = poly_2d.T
            z_start = max(0, slice_idx - half_t)
            z_end = min(dim_z, slice_idx + half_t + (thickness % 2))
            for z in range(z_start, z_end):
                if mode == "add":
                    new_mask[:, :, z] = np.logical_or(new_mask[:, :, z], poly_xy).astype(np.uint8)
                elif mode == "replace":
                    new_mask[:, :, z] = poly_xy
                elif mode == "intersect":
                    new_mask[:, :, z] = np.logical_and(new_mask[:, :, z], poly_xy).astype(np.uint8)

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"连续多边形绘制 (平面={plane}, 切片={slice_idx}, 包含点={len(points)}个)"
        )


class ContourScissorsCutTool(BaseMedicalTool):
    """连续曲线多边形剪刀裁切算子 (Contour Scissors Cut / Freehand Cut)"""
    @property
    def name(self) -> str:
        return "contour_scissors_cut"

    @property
    def description(self) -> str:
        return (
            "【原子工具: 连续多边形剪刀裁切】在指定切片方向上，使用连续闭合多边形曲线作为剪刀路径进行精准解剖剥离。"
            "可选择 'remove_inside'(剪掉多边形内部粘连组织) 或 'remove_outside'(仅保留多边形内部，剪除外部所有杂质)。"
            "彻底解决单轴平切导致的直角方形边缘问题！"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "裁切参考平面: 'sagittal'(矢状位), 'coronal'(冠状位), 'axial'(轴位)。"
                },
                "slice_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "适用的切片范围 [start_idx, end_idx]，不传则默认沿该方向贯穿全图裁切。"
                },
                "polygon_points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "剪刀切割多边形闭合控制点序列 [[c1, c2], ...]。"
                },
                "cut_mode": {
                    "type": "string",
                    "enum": ["remove_inside", "remove_outside"],
                    "description": "'remove_inside'(剪除多边形内部粘连组织), 'remove_outside'(保留内部，剪除外部所有软组织)。默认 'remove_inside'。"
                }
            },
            "required": ["plane", "polygon_points"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask.copy()
        dim_x, dim_y, dim_z = context.shape
        plane = kwargs.get("plane", "sagittal").lower()
        slice_range = kwargs.get("slice_range", None)
        points = kwargs.get("polygon_points", [])
        cut_mode = kwargs.get("cut_mode", "remove_inside")

        if len(points) < 3:
            return ToolResult(success=False, new_mask=old_mask, message="剪刀多边形点数不足 3 个，无法构成裁切区域")

        from .vtk_engine import VTKSegmentationEngine
        new_mask = VTKSegmentationEngine.apply_polygon_stencil_scissors(
            mask_3d=old_mask,
            plane=plane,
            points_2d=points,
            slice_range=slice_range,
            cut_mode=cut_mode,
            spacing=context.spacing
        )

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"多边形曲线剪刀 (模式={cut_mode}, 平面={plane}, 包含点={len(points)}个)"
        )


class MarkerControlledWatershedTool(BaseMedicalTool):
    """【分水岭解剖图割算子 (Marker-Controlled Watershed / GrowCut)】"""
    @property
    def name(self) -> str:
        return "watershed_segmentation"

    @property
    def description(self) -> str:
        return (
            "【高级原子工具: 分水岭解剖图割 (Watershed / GrowCut)】仿照 3D Slicer GrowCut 与 Photoshop 快速选择。"
            "只需提供前景种子点 (foreground_points, 目标器官内部) 与背景种子点 (background_points, 周围非目标/脑干/水腔/颅骨)，"
            "系统沿影像三维梯度幅值地形图进行自适应水流漫延与边缘阻断，自动生成极度贴合自然解剖纹理的掩码！"
            "支持 2D 切片输入 (plane + slice_index + fg_points_2d + bg_points_2d) 或 3D 空间坐标序列。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "可选: 切片平面 ('axial', 'coronal', 'sagittal')。"
                },
                "slice_index": {
                    "type": "integer",
                    "description": "可选: 切片绝对索引。"
                },
                "fg_points_2d": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "可选: 切片二维前景种子点序列 [[c1, c2], ...] (目标组织内部)。"
                },
                "bg_points_2d": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "可选: 切片二维背景种子点序列 [[c1, c2], ...] (周围不需要的组织/脑干/脑脊液)。"
                },
                "foreground_points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "可选: 三维前景种子点序列 [[x, y, z], ...]。"
                },
                "background_points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "可选: 三维背景种子点序列 [[x, y, z], ...]。"
                },
                "compactness": {
                    "type": "number",
                    "description": "分水岭平滑紧凑度因子 (0.001 ~ 0.5，默认 0.05)。"
                },
                "mode": {
                    "type": "string",
                    "enum": ["replace", "add", "intersect"],
                    "description": "掩码更新模式: 'replace'(替换), 'add'(合并), 'intersect'(求交)。默认 'replace'。"
                }
            }
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask.copy()
        img = context.image_data.astype(np.float32)
        dim_x, dim_y, dim_z = context.shape
        compactness = float(kwargs.get("compactness", 0.05))
        mode = kwargs.get("mode", "replace")

        # 解析前景点与背景点
        fg_3d = []
        bg_3d = []

        if "foreground_points" in kwargs:
            for p in kwargs["foreground_points"]:
                if len(p) == 3:
                    fg_3d.append([int(p[0]), int(p[1]), int(p[2])])
        if "background_points" in kwargs:
            for p in kwargs["background_points"]:
                if len(p) == 3:
                    bg_3d.append([int(p[0]), int(p[1]), int(p[2])])

        # 2D 切片投影点解析
        plane = kwargs.get("plane", "").lower()
        s_idx = kwargs.get("slice_index", None)
        fg_2d = kwargs.get("fg_points_2d", [])
        bg_2d = kwargs.get("bg_points_2d", [])

        if plane and s_idx is not None:
            s_idx = int(s_idx)
            for pt in fg_2d:
                if len(pt) >= 2:
                    c1, c2 = int(round(pt[0])), int(round(pt[1]))
                    if plane == "sagittal":
                        fg_3d.append([np.clip(s_idx, 0, dim_x - 1), np.clip(c1, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)])
                    elif plane == "coronal":
                        fg_3d.append([np.clip(c1, 0, dim_x - 1), np.clip(s_idx, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)])
                    elif plane == "axial":
                        fg_3d.append([np.clip(c1, 0, dim_x - 1), np.clip(c2, 0, dim_y - 1), np.clip(s_idx, 0, dim_z - 1)])

            for pt in bg_2d:
                if len(pt) >= 2:
                    c1, c2 = int(round(pt[0])), int(round(pt[1]))
                    if plane == "sagittal":
                        bg_3d.append([np.clip(s_idx, 0, dim_x - 1), np.clip(c1, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)])
                    elif plane == "coronal":
                        bg_3d.append([np.clip(c1, 0, dim_x - 1), np.clip(s_idx, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)])
                    elif plane == "axial":
                        bg_3d.append([np.clip(c1, 0, dim_x - 1), np.clip(c2, 0, dim_y - 1), np.clip(s_idx, 0, dim_z - 1)])

        if not fg_3d:
            return ToolResult(success=False, new_mask=old_mask, message="分水岭算法失败: 未提供有效的前景种子点 (foreground_points)")

        # 1. 计算三维空间梯度幅值 (Gradient Magnitude) 作为分水岭地形图
        smoothed = ndimage.gaussian_filter(img, sigma=1.0)
        gradient = ndimage.gaussian_gradient_magnitude(smoothed, sigma=1.0)

        # 2. 建立 Markers 标记矩阵: 1=前景, 2=背景, 0=未知
        markers = np.zeros((dim_x, dim_y, dim_z), dtype=np.int32)
        for fx, fy, fz in fg_3d:
            markers[max(0, fx-1):min(dim_x, fx+2), max(0, fy-1):min(dim_y, fy+2), max(0, fz-1):min(dim_z, fz+2)] = 1

        if bg_3d:
            for bx, by, bz in bg_3d:
                markers[max(0, bx-1):min(dim_x, bx+2), max(0, by-1):min(dim_y, by+2), max(0, bz-1):min(dim_z, bz+2)] = 2
        else:
            # 自动在图像极边缘标记背景
            markers[0, :, :] = 2
            markers[-1, :, :] = 2
            markers[:, 0, :] = 2
            markers[:, -1, :] = 2
            markers[:, :, 0] = 2
            markers[:, :, -1] = 2

        # 3. 运行分水岭分割
        try:
            from skimage.segmentation import watershed
            ws_labels = watershed(gradient, markers=markers, compactness=compactness)
            extracted = (ws_labels == 1).astype(np.uint8)
        except Exception:
            # 回退实现: 基于测地线距离的最短路径分水岭
            dist_fg = ndimage.distance_transform_edt(markers != 1)
            dist_bg = ndimage.distance_transform_edt(markers != 2)
            extracted = (dist_fg < dist_bg).astype(np.uint8)

        # 孔洞充填与微闭运算
        struct = ndimage.generate_binary_structure(3, 1)
        extracted = ndimage.binary_fill_holes(extracted).astype(np.uint8)
        extracted = ndimage.binary_closing(extracted, structure=struct, iterations=1).astype(np.uint8)

        if mode == "add":
            new_mask = np.logical_or(old_mask, extracted).astype(np.uint8)
        elif mode == "intersect":
            new_mask = np.logical_and(old_mask, extracted).astype(np.uint8)
        else:
            new_mask = extracted

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["fg_points_count"] = len(fg_3d)
        metrics["bg_points_count"] = len(bg_3d)

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"分水岭自适应图割 (前景点={len(fg_3d)}, 背景点={len(bg_3d)}, 模式={mode})"
        )


class SmartIntensityBrushTool(BaseMedicalTool):
    """【PS智能边缘吸附画笔 (Smart Edge-Aware Paint Brush)】"""
    @property
    def name(self) -> str:
        return "smart_intensity_brush"

    @property
    def description(self) -> str:
        return (
            "【高级原子工具: PS智能吸附画笔 (Smart Edge-Aware Brush)】仿照 Photoshop 智能快速选区画笔与磁性套索。"
            "在指定切片或三维空间涂抹时，以笔触中心点处的组织信号为基准，自动在该半径范围内【仅吸收同类组织信号】并在遇解剖边界/脑脊液/颅骨时自动智能吸附贴边阻断！"
            "支持直接传入切片二维坐标 (plane + slice_index + point_2d) 或三维空间坐标 (center)。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "可选: 切片平面 ('axial', 'coronal', 'sagittal')。"
                },
                "slice_index": {
                    "type": "integer",
                    "description": "可选: 切片绝对索引 (如 Z=35, X=91)。"
                },
                "point_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "可选: 切片二维坐标点 [c1, c2]。"
                },
                "center": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "可选: 画笔球心三维坐标 [x, y, z]。"
                },
                "radius_mm": {
                    "type": "number",
                    "description": "画笔物理作用半径 (毫米 mm，如 6.0, 10.0, 15.0)。默认 8.0。"
                },
                "intensity_tolerance": {
                    "type": "number",
                    "description": "组织信号吸附容差范围 (如 600 ~ 2500，默认自适应计算)。"
                },
                "mode": {
                    "type": "string",
                    "enum": ["add", "erase", "replace"],
                    "description": "画笔模式: 'add'(智能吸附涂抹), 'erase'(智能吸附擦除), 'replace'(替换)。默认 'add'。"
                }
            }
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask.copy()
        img = context.image_data.astype(np.float32)
        dim_x, dim_y, dim_z = context.shape
        sp_x, sp_y, sp_z = context.spacing
        radius_mm = float(kwargs.get("radius_mm", 8.0))
        mode = kwargs.get("mode", "add")

        center = None
        if "center" in kwargs and len(kwargs["center"]) == 3:
            center = [int(kwargs["center"][0]), int(kwargs["center"][1]), int(kwargs["center"][2])]

        plane = kwargs.get("plane", "").lower()
        s_idx = kwargs.get("slice_index", None)
        p2d = kwargs.get("point_2d", None)

        if plane and s_idx is not None and p2d and len(p2d) >= 2:
            s_idx = int(s_idx)
            c1, c2 = int(round(p2d[0])), int(round(p2d[1]))
            if plane == "sagittal":
                center = [np.clip(s_idx, 0, dim_x - 1), np.clip(c1, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)]
            elif plane == "coronal":
                center = [np.clip(c1, 0, dim_x - 1), np.clip(s_idx, 0, dim_y - 1), np.clip(c2, 0, dim_z - 1)]
            elif plane == "axial":
                center = [np.clip(c1, 0, dim_x - 1), np.clip(c2, 0, dim_y - 1), np.clip(s_idx, 0, dim_z - 1)]

        if center is None:
            center = [dim_x // 2, dim_y // 2, dim_z // 2]

        cx, cy, cz = center
        rx = max(1, int(round(radius_mm / sp_x)))
        ry = max(1, int(round(radius_mm / sp_y)))
        rz = max(1, int(round(radius_mm / sp_z)))

        x0, x1 = max(0, cx - rx), min(dim_x, cx + rx + 1)
        y0, y1 = max(0, cy - ry), min(dim_y, cy + ry + 1)
        z0, z1 = max(0, cz - rz), min(dim_z, cz + rz + 1)

        # 1. 提取画笔中心邻域的基准组织强度均值与方差
        seed_patch = img[max(0, cx-1):min(dim_x, cx+2), max(0, cy-1):min(dim_y, cy+2), max(0, cz-1):min(dim_z, cz+2)]
        seed_mean = float(np.mean(seed_patch))
        seed_std = float(np.std(seed_patch))

        tol = kwargs.get("intensity_tolerance", None)
        if tol is None or tol <= 0:
            tol = max(800.0, seed_std * 2.8)
        else:
            tol = float(tol)

        # 2. 在球形 ROI 内进行强度敏感筛选与连通域生长
        roi_img = img[x0:x1, y0:y1, z0:z1]
        x_grid, y_grid, z_grid = np.ogrid[x0-cx:x1-cx, y0-cy:y1-cy, z0-cz:z1-cz]
        sphere_patch = ((x_grid * sp_x)**2 + (y_grid * sp_y)**2 + (z_grid * sp_z)**2) <= (radius_mm**2)

        intensity_match = (np.abs(roi_img - seed_mean) <= tol)
        candidate = np.logical_and(sphere_patch, intensity_match).astype(np.uint8)

        # 3. 仅保留与中心种子相连通的区域 (自动被高梯度边缘/脑脊液阻断)
        local_seed_x = cx - x0
        local_seed_y = cy - y0
        local_seed_z = cz - z0
        labeled, num_f = ndimage.label(candidate)
        seed_label = labeled[local_seed_x, local_seed_y, local_seed_z]
        if seed_label > 0:
            smart_patch = (labeled == seed_label).astype(np.uint8)
        else:
            smart_patch = candidate

        smart_total = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
        smart_total[x0:x1, y0:y1, z0:z1] = smart_patch

        if mode == "erase":
            new_mask = np.logical_and(old_mask, np.logical_not(smart_total)).astype(np.uint8)
        elif mode == "replace":
            new_mask = smart_total
        else:
            new_mask = np.logical_or(old_mask, smart_total).astype(np.uint8)

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["smart_center"] = [cx, cy, cz]
        metrics["seed_intensity_mean"] = seed_mean
        metrics["intensity_tolerance"] = tol

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"PS智能吸附画笔 (中心={center}, 半径={radius_mm}mm, 容差={tol:.1f}, 模式={mode})"
        )


