import cv2
import base64
import numpy as np
from typing import Dict, Any, List, Optional
from .base_tool import BaseMedicalTool, ImageContext, ToolResult


class InspectOrthoSliceTool(BaseMedicalTool):
    """
    【放射学主动视觉探针】主动调取并审视任意指定正交断层切片 (Axial / Coronal / Sagittal)
    支持任意切片索引跳转与局部 ROI 高清变焦放大 (Zoom-In)。
    """
    @property
    def name(self) -> str:
        return "inspect_ortho_slice"

    @property
    def description(self) -> str:
        return (
            "【视觉探针算子: 切片跳转与主动审视】像放射科医生滚动鼠标滚轮一样，主动调取并跳转到任意指定的正交切片断面。"
            "系统会立即将该切片的最新叠加图像喂入你的多模态视觉上下文中！"
            "支持局部放大变焦 (zoom_roi)，看清微小叶段白质树与解剖边界细节。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "要审视的切片视角: 'axial'(轴位横断面), 'coronal'(冠状位额状面), 'sagittal'(矢状面, 如正中矢状面 X=91)。"
                },
                "slice_index": {
                    "type": "integer",
                    "description": "要跳转查看的切片绝对索引 (如矢状面 X=91, 轴位 Z=35, 冠状位 Y=60 等)。"
                },
                "zoom_roi": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "可选的局部放大变焦区域 [c1_min, c2_min, c1_max, c2_max]，对目标器官进行 1:1 亚体素高清特写放大。"
                },
                "overlay_mask": {
                    "type": "boolean",
                    "description": "是否以青色半透明叠加当前已标定的 Mask。默认 True。"
                }
            },
            "required": ["plane", "slice_index"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        dim_x, dim_y, dim_z = context.shape
        plane = kwargs.get("plane", "sagittal").lower()
        slice_idx = int(kwargs.get("slice_index", 0))
        zoom_roi = kwargs.get("zoom_roi", None)
        overlay_mask = kwargs.get("overlay_mask", True)
        
        img = context.image_data
        mask = context.current_mask if overlay_mask else None
        
        # 窗宽窗位归一化
        wl = 3500.0
        ww = 7000.0
        min_v = wl - (ww / 2.0)
        rng = max(1.0, ww)

        if plane == "sagittal":
            slice_idx = np.clip(slice_idx, 0, dim_x - 1)
            raw_slice = np.rot90(img[slice_idx, :, :], 1)
            raw_m = np.rot90(mask[slice_idx, :, :], 1) if mask is not None else None
            plane_label = f"Sagittal 矢状面 X={slice_idx}"
        elif plane == "coronal":
            slice_idx = np.clip(slice_idx, 0, dim_y - 1)
            raw_slice = np.rot90(img[:, slice_idx, :], 1)
            raw_m = np.rot90(mask[:, slice_idx, :], 1) if mask is not None else None
            plane_label = f"Coronal 冠状面 Y={slice_idx}"
        else:  # axial
            slice_idx = np.clip(slice_idx, 0, dim_z - 1)
            raw_slice = np.rot90(img[:, :, slice_idx], 1)
            raw_m = np.rot90(mask[:, :, slice_idx], 1) if mask is not None else None
            plane_label = f"Axial 轴位断面 Z={slice_idx}"

        clipped = np.clip(raw_slice, min_v, min_v + rng)
        norm = ((clipped - min_v) / rng * 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

        if raw_m is not None and np.any(raw_m > 0):
            over = rgb.copy()
            over[raw_m > 0] = [212, 182, 6]  # 青色高亮
            rgb = cv2.addWeighted(over, 0.50, rgb, 0.50, 0)

        # 局部变焦裁剪 (Zoom-In)
        if zoom_roi and len(zoom_roi) == 4:
            h, w = rgb.shape[:2]
            c1_min, c2_min, c1_max, c2_max = zoom_roi
            r0 = max(0, min(c2_min, c2_max))
            r1 = min(h, max(c2_min, c2_max))
            c0 = max(0, min(c1_min, c1_max))
            c1 = min(w, max(c1_min, c1_max))
            if r1 > r0 + 5 and c1 > c0 + 5:
                rgb = rgb[r0:r1, c0:c1]
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                plane_label += f" [局部高清变焦 放大区域: {zoom_roi}]"

        cv2.putText(rgb, plane_label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # 编码为 JPEG Base64
        _, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_str = base64.b64encode(buf).decode('utf-8')

        # 计算该切片层内部的统计数据
        slice_mask_voxels = int(np.count_nonzero(raw_m > 0)) if raw_m is not None else 0
        
        metrics = self.compute_metrics(context.current_mask, context.current_mask, context)
        metrics["inspected_plane"] = str(plane)
        metrics["inspected_slice_index"] = int(slice_idx)
        metrics["slice_mask_voxels"] = int(slice_mask_voxels)

        # 在 ToolResult 中附带生成的图像部件
        image_part = {
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": b64_str
            }
        }

        return ToolResult(
            success=True,
            new_mask=context.current_mask,
            observation_metrics=metrics,
            action_description=f"主动调取断层图像: {plane_label} (当前层 Mask 像素量: {slice_mask_voxels})",
            message=f"已成功为您跳转并渲染最新断层切片: {plane_label}。图像已注入当前对话视觉上下文中，请观察解剖轮廓与边缘粘连！",
            attached_image_part=image_part
        )


class BrowseSliceGalleryTool(BaseMedicalTool):
    """
    【局部连续切片画廊发生器】沿指定视角生成连续 N 层断层的画廊矩阵
    """
    @property
    def name(self) -> str:
        return "browse_slice_gallery"

    @property
    def description(self) -> str:
        return (
            "【视觉探针算子: 连续断层画廊浏览】沿指定方向 (如后颅窝小脑 Z=15~65 或矢状面 X=70~110) 连续抓取一组切片并拼贴为多格画廊。"
            "让你一次性俯瞰目标器官在三维空间中的完整连续解剖演变！"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plane": {
                    "type": "string",
                    "enum": ["axial", "coronal", "sagittal"],
                    "description": "切片视角: 'axial', 'coronal', 'sagittal'。"
                },
                "start_slice": {
                    "type": "integer",
                    "description": "起始切片索引 (如 20)"
                },
                "end_slice": {
                    "type": "integer",
                    "description": "终止切片索引 (如 60)"
                },
                "step": {
                    "type": "integer",
                    "description": "切片采样间隔步长 (如 5，默认自动适配 4~8 张切片)"
                }
            },
            "required": ["plane", "start_slice", "end_slice"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        dim_x, dim_y, dim_z = context.shape
        plane = kwargs.get("plane", "axial").lower()
        s0 = max(0, int(kwargs.get("start_slice", 0)))
        s1 = int(kwargs.get("end_slice", 50))
        step = max(1, int(kwargs.get("step", 5)))

        img = context.image_data
        mask = context.current_mask

        wl = 3500.0
        ww = 7000.0
        min_v = wl - (ww / 2.0)
        rng = max(1.0, ww)

        max_bound = dim_x if plane == "sagittal" else (dim_y if plane == "coronal" else dim_z)
        s1 = min(max_bound - 1, s1)
        if s1 <= s0:
            s1 = min(max_bound - 1, s0 + 10)

        indices = list(range(s0, s1 + 1, step))[:8]  # 最多取 8 张拼接
        tiles = []

        for idx in indices:
            if plane == "sagittal":
                raw_s = np.rot90(img[idx, :, :], 1)
                raw_m = np.rot90(mask[idx, :, :], 1) if mask is not None else None
            elif plane == "coronal":
                raw_s = np.rot90(img[:, idx, :], 1)
                raw_m = np.rot90(mask[:, idx, :], 1) if mask is not None else None
            else:
                raw_s = np.rot90(img[:, :, idx], 1)
                raw_m = np.rot90(mask[:, :, idx], 1) if mask is not None else None

            clipped = np.clip(raw_s, min_v, min_v + rng)
            norm = ((clipped - min_v) / rng * 255.0).astype(np.uint8)
            rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

            if raw_m is not None and np.any(raw_m > 0):
                over = rgb.copy()
                over[raw_m > 0] = [212, 182, 6]
                rgb = cv2.addWeighted(over, 0.50, rgb, 0.50, 0)

            cv2.putText(rgb, f"{plane[0].upper()}={idx}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(rgb)

        sheet = np.hstack(tiles) if tiles else np.zeros((100, 100, 3), dtype=np.uint8)

        _, buf = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_str = base64.b64encode(buf).decode('utf-8')

        image_part = {
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": b64_str
            }
        }

        metrics = self.compute_metrics(context.current_mask, context.current_mask, context)
        metrics["browsed_slices"] = [int(x) for x in indices]

        return ToolResult(
            success=True,
            new_mask=context.current_mask,
            observation_metrics=metrics,
            action_description=f"生成连续断层画廊: {plane} (索引={indices})",
            message=f"已成功生成 {plane} 方向连续 {len(indices)} 层断层切片画廊，图像已注入视觉上下文中！",
            attached_image_part=image_part
        )
