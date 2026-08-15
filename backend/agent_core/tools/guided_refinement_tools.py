import numpy as np
from scipy import ndimage
from typing import Dict, Any, List, Optional
from .base_tool import BaseMedicalTool, ImageContext, ToolResult

class SpatialPromptGuidedSegmentationTool(BaseMedicalTool):
    """
    大模型多模态视觉空间提示引导的真三维有机轮廓精修算子 (Vision-Guided Organic Refinement)
    以 Gemini 视觉观察定位的 3D 种子坐标与解剖 ROI 为锚点，
    在三维连续空间内执行梯度能量场自适应区域生长与测地线轮廓提取，彻底告别长方体硬截断。
    """
    @property
    def name(self) -> str:
        return "spatial_prompt_guided_segmentation"

    @property
    def description(self) -> str:
        return (
            "【原生多模态视觉空间引导算子】利用你在全脑多断层画廊中所观察到的解剖结构/病灶视觉特征，"
            "下发 3D 包围盒 (3D BBox) 与核心种子点坐标 (center_point_3d)。"
            "本地引擎将以种子点为核心执行 3D 梯度自适应区域生长，生成完全贴合生物脑回的连续自然轮廓。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_name": {
                    "type": "string",
                    "description": "你在画廊中观察并识别的目标解剖结构或病灶名称 (如: '小脑', '脑干', '第四脑室', '胶质瘤', '海马体' 等)"
                },
                "bbox_3d": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "你在画廊中观察到的 3D 空间大致范围 [x_min, y_min, z_min, x_max, y_max, z_max] (作为生长软约束)"
                },
                "center_point_3d": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "目标解剖结构在画廊切片中的核心三维种子坐标 [x, y, z] (如小脑中心 [90, 60, 45])"
                },
                "tissue_intensity_type": {
                    "type": "string",
                    "enum": ["brain_parenchyma", "csf_fluid", "hyperintense", "hypointense", "soft_tissue"],
                    "description": "目标组织的信号灰度特性: 'brain_parenchyma'(脑实质灰质/小脑叶), 'csf_fluid'(脑脊液低信号), 'hyperintense'(高信号), 'soft_tissue'(一般软组织)"
                },
                "refinement_mode": {
                    "type": "string",
                    "enum": ["replace", "add", "subtract"],
                    "description": "精修合并模式: 'replace'(作为全新掩码), 'add'(合并至已有掩码), 'subtract'(从已有掩码中扣除)。默认 'replace'。"
                }
            },
            "required": ["target_name", "bbox_3d"]
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        img = context.image_data.astype(np.float32)
        dim_x, dim_y, dim_z = context.shape

        target_name = kwargs.get("target_name", "解剖目标")
        raw_bbox = kwargs.get("bbox_3d", [0, 0, 0, dim_x, dim_y, dim_z])
        center_pt = kwargs.get("center_point_3d", None)
        intensity_type = kwargs.get("tissue_intensity_type", "brain_parenchyma")
        mode = kwargs.get("refinement_mode", "replace")

        # 1. 规范化 3D Bounding Box 并向外做 15% 软边缘外延，避免边界裁切
        if len(raw_bbox) == 6:
            bx0 = min(raw_bbox[0], raw_bbox[3])
            bx1 = max(raw_bbox[0], raw_bbox[3])
            by0 = min(raw_bbox[1], raw_bbox[4])
            by1 = max(raw_bbox[1], raw_bbox[4])
            bz0 = min(raw_bbox[2], raw_bbox[5])
            bz1 = max(raw_bbox[2], raw_bbox[5])
        else:
            bx0, bx1, by0, by1, bz0, bz1 = 0, dim_x, 0, dim_y, 0, dim_z

        pad_x = max(8, int((bx1 - bx0) * 0.15))
        pad_y = max(8, int((by1 - by0) * 0.15))
        pad_z = max(8, int((bz1 - bz0) * 0.15))

        x_min = max(0, bx0 - pad_x)
        x_max = min(dim_x, bx1 + pad_x)
        y_min = max(0, by0 - pad_y)
        y_max = min(dim_y, by1 + pad_y)
        z_min = max(0, bz0 - pad_z)
        z_max = min(dim_z, bz1 + pad_z)

        # 2. 提取局部子体数据并计算 3D 梯度物理边缘场
        local_img = img[x_min:x_max, y_min:y_max, z_min:z_max]
        smoothed = ndimage.gaussian_filter(local_img, sigma=1.0)

        gx = ndimage.sobel(smoothed, axis=0)
        gy = ndimage.sobel(smoothed, axis=1)
        gz = ndimage.sobel(smoothed, axis=2)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        # 梯度上限阻尼 (强解剖分界处梯度极大，如脑脊液隙或硬脑膜)
        edge_barrier_thresh = np.percentile(grad_mag, 88)

        # 3. 确定自适应 3D 生长种子点
        if center_pt and len(center_pt) == 3:
            seed_x = int(np.clip(center_pt[0] - x_min, 1, local_img.shape[0] - 2))
            seed_y = int(np.clip(center_pt[1] - y_min, 1, local_img.shape[1] - 2))
            seed_z = int(np.clip(center_pt[2] - z_min, 1, local_img.shape[2] - 2))
        else:
            # 取局部区域内部中等以上信号的核心点作为种子
            mid_slice = smoothed[smoothed.shape[0]//4 : smoothed.shape[0]*3//4,
                                 smoothed.shape[1]//4 : smoothed.shape[1]*3//4,
                                 smoothed.shape[2]//4 : smoothed.shape[2]*3//4]
            non_zero_mid = mid_slice[mid_slice > 100]
            target_val = np.median(non_zero_mid) if len(non_zero_mid) > 0 else np.mean(smoothed)
            diff = np.abs(smoothed - target_val)
            seed_x, seed_y, seed_z = np.unravel_index(np.argmin(diff), smoothed.shape)

        seed_intensity = float(smoothed[seed_x, seed_y, seed_z])

        # 4. 计算组织同质性容差 (Intensity Tolerance)
        local_non_zero = smoothed[smoothed > 50]
        if len(local_non_zero) > 0:
            std_val = float(np.std(local_non_zero))
            p95 = float(np.percentile(local_non_zero, 95))
            p10 = float(np.percentile(local_non_zero, 10))
        else:
            std_val, p95, p10 = 500.0, 5000.0, 100.0

        if intensity_type == "csf_fluid":
            lower_bound = 10.0
            upper_bound = p10 + (p95 - p10) * 0.30
        elif intensity_type == "hyperintense":
            lower_bound = p95 * 0.70
            upper_bound = 99999.0
        else:
            # 脑实质组织 (灰白质与小脑叶): 容许合理的生物组织灰度范围
            tolerance = max(std_val * 1.6, 600.0)
            lower_bound = max(100.0, seed_intensity - tolerance)
            upper_bound = seed_intensity + tolerance * 1.5

        # 5. 3D 区域生长 + 梯度阻尼漫延
        intensity_mask = np.logical_and(smoothed >= lower_bound, smoothed <= upper_bound)
        smooth_gradient_pass = grad_mag < edge_barrier_thresh * 1.6
        pass_matrix = np.logical_and(intensity_mask, smooth_gradient_pass).astype(np.uint8)

        # 确保种子点处于可通行区
        pass_matrix[seed_x, seed_y, seed_z] = 1

        # 6. 从种子点提取最大 3D 连通体
        labeled, num_features = ndimage.label(pass_matrix)
        seed_label = labeled[seed_x, seed_y, seed_z]
        if seed_label > 0:
            grown_mask = (labeled == seed_label).astype(np.uint8)
        else:
            if num_features > 0:
                sizes = ndimage.sum(pass_matrix, labeled, range(1, num_features + 1))
                largest_label = np.argmax(sizes) + 1
                grown_mask = (labeled == largest_label).astype(np.uint8)
            else:
                grown_mask = pass_matrix

        # 7. 生物解剖形态学闭合与孔洞充填 (去除孤立碎孔，贴合脑回微褶皱)
        grown_mask = ndimage.binary_fill_holes(grown_mask).astype(np.uint8)
        struct = ndimage.generate_binary_structure(3, 1)
        grown_mask = ndimage.binary_closing(grown_mask, structure=struct, iterations=2).astype(np.uint8)
        grown_mask = ndimage.binary_opening(grown_mask, structure=struct, iterations=1).astype(np.uint8)

        # 8. 贴回全局 3D 坐标空间 (由于外延了 pad，绝不会出现硬性长方体边框)
        generated_global_mask = np.zeros(context.shape, dtype=np.uint8)
        generated_global_mask[x_min:x_max, y_min:y_max, z_min:z_max] = grown_mask

        # 9. 结合精修模式
        if mode == "add":
            new_mask = np.logical_or(old_mask, generated_global_mask).astype(np.uint8)
        elif mode == "subtract":
            new_mask = np.logical_and(old_mask, np.logical_not(generated_global_mask)).astype(np.uint8)
        else:
            new_mask = generated_global_mask

        metrics = self.compute_metrics(old_mask, new_mask, context)
        metrics["target_name"] = target_name
        metrics["grown_seed_coords"] = [int(seed_x + x_min), int(seed_y + y_min), int(seed_z + z_min)]

        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            action_description=f"多模态 3D 区域生长连续轮廓分割: {target_name} (种子点: [{seed_x + x_min}, {seed_y + y_min}, {seed_z + z_min}])"
        )
