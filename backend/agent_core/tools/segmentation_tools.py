import numpy as np
from scipy import ndimage
from typing import Dict, Any, Optional
from .base_tool import BaseMedicalTool, ImageContext, ToolResult

class BrainTissueExtractionTool(BaseMedicalTool):
    """自适应脑实质与解剖亚结构 (小脑/脑干/脑室/灰白质) 分割提取算子"""
    @property
    def name(self) -> str:
        return "extract_brain_tissue"

    @property
    def description(self) -> str:
        return (
            "自适应提取头颅 MRI 中的脑组织及解剖亚结构并去除颅骨与眼眶杂质。"
            "支持目标区域: 'all' (全脑实质), 'left_hemisphere' (左大脑半球), 'right_hemisphere' (右大脑半球), "
            "'cerebellum' (小脑), 'brainstem' (脑干), 'ventricles' (侧脑室), 'white_matter' (脑白质), 'gray_matter_cortex' (大脑皮质灰质)。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "enum": [
                        "all", 
                        "left_hemisphere", 
                        "right_hemisphere", 
                        "cerebellum", 
                        "brainstem", 
                        "ventricles", 
                        "white_matter", 
                        "gray_matter_cortex"
                    ],
                    "description": "目标解剖区域: 'all' (全脑), 'cerebellum' (小脑), 'brainstem' (脑干), 'ventricles' (脑室), 'left_hemisphere' (左半球), 'right_hemisphere' (右半球), 'white_matter' (白质), 'gray_matter_cortex' (大脑皮质)。默认为 'all'。"
                },
                "threshold_ratio": {
                    "type": "number",
                    "description": "背景与组织分离的相对强度阈值比例 (0.1 ~ 0.5)，默认 0.22。"
                }
            },
            "required": []
        }

    def execute(self, context: ImageContext, **kwargs) -> ToolResult:
        old_mask = context.current_mask
        img = context.image_data
        region = kwargs.get("region", "all")
        ratio = kwargs.get("threshold_ratio", 0.22)
        dim_x, dim_y, dim_z = context.shape

        # 1. 估算组织强度分布 (自适应高斯平滑降噪)
        smoothed = ndimage.gaussian_filter(img.astype(np.float32), sigma=1.2)
        non_zero = smoothed[smoothed > 10]
        if len(non_zero) == 0:
            threshold = float(np.mean(img))
        else:
            p98 = np.percentile(non_zero, 98)
            threshold = p98 * ratio

        # 2. 粗提取颅内组织
        binary = (smoothed > threshold).astype(np.uint8)

        # 3. 三维连通域提取最大颅内主体
        labeled, num_features = ndimage.label(binary)
        if num_features > 0:
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            brain_mask = (labeled == largest_label).astype(np.uint8)
        else:
            brain_mask = binary

        # 4. 孔洞填充与轻微闭运算平滑边缘
        brain_mask = ndimage.binary_fill_holes(brain_mask).astype(np.uint8)
        struct = ndimage.generate_binary_structure(3, 1)
        brain_mask = ndimage.binary_closing(brain_mask, structure=struct, iterations=1).astype(np.uint8)

        # 5. 解剖亚结构空间与概率先验切分 (MNI-Space Anatomical Priors)
        if region in ["left_hemisphere", "right_hemisphere"]:
            mid_x = dim_x // 2
            hemisphere_mask = np.zeros_like(brain_mask)
            if region == "left_hemisphere":
                hemisphere_mask[:mid_x, :, :] = 1
                desc = "左大脑半球实质提取"
            else:
                hemisphere_mask[mid_x:, :, :] = 1
                desc = "右大脑半球实质提取"
            new_mask = np.logical_and(brain_mask, hemisphere_mask).astype(np.uint8)

        elif region == "cerebellum":
            # 小脑位于颅后窝 (Posterior Fossa)，在解剖空间中处于下部 (Z < 0.45 * dim_z) 且偏后侧 (Y < 0.55 * dim_y)
            cerebellum_roi = np.zeros_like(brain_mask)
            z_limit = int(dim_z * 0.46)
            y_limit = int(dim_y * 0.55)
            x_margin = int(dim_x * 0.12)
            cerebellum_roi[x_margin:dim_x - x_margin, :y_limit, :z_limit] = 1
            
            raw_cerebellum = np.logical_and(brain_mask, cerebellum_roi).astype(np.uint8)
            # 提取最大连通域得到完整小脑
            cb_labeled, cb_num = ndimage.label(raw_cerebellum)
            if cb_num > 0:
                cb_sizes = ndimage.sum(raw_cerebellum, cb_labeled, range(1, cb_num + 1))
                largest_cb = np.argmax(cb_sizes) + 1
                new_mask = (cb_labeled == largest_cb).astype(np.uint8)
                new_mask = ndimage.binary_fill_holes(new_mask).astype(np.uint8)
            else:
                new_mask = raw_cerebellum
            desc = "小脑 (Cerebellum) 解剖亚结构精确定位与提取"

        elif region == "brainstem":
            # 脑干位于中线柱状区域 (Z 轴下部 0.15~0.55，中线 X 核心与 Y 中央)
            brainstem_roi = np.zeros_like(brain_mask)
            z_min = int(dim_z * 0.12)
            z_max = int(dim_z * 0.52)
            y_min = int(dim_y * 0.40)
            y_max = int(dim_y * 0.65)
            x_min = int(dim_x * 0.38)
            x_max = int(dim_x * 0.62)
            brainstem_roi[x_min:x_max, y_min:y_max, z_min:z_max] = 1
            new_mask = np.logical_and(brain_mask, brainstem_roi).astype(np.uint8)
            desc = "脑干 (Brainstem) 解剖中枢定位与提取"

        elif region == "ventricles":
            # 脑室为脑实质内部的低信号脑脊液 (CSF) 腔隙
            csf_threshold = p98 * 0.45
            internal_roi = ndimage.binary_erosion(brain_mask, iterations=8).astype(np.uint8)
            low_signal = np.logical_and(smoothed < csf_threshold, smoothed > 10).astype(np.uint8)
            vent_candidate = np.logical_and(internal_roi, low_signal).astype(np.uint8)
            new_mask = ndimage.binary_opening(vent_candidate, iterations=1).astype(np.uint8)
            desc = "侧脑室与中央脑脊液腔隙 (Ventricles) 提取"

        elif region == "white_matter":
            # 脑白质为 T1w 图像中的高信号脑髓质部分 (强度大于 0.68 * p98)
            wm_threshold = p98 * 0.65
            wm_candidate = np.logical_and(brain_mask, smoothed >= wm_threshold).astype(np.uint8)
            new_mask = ndimage.binary_opening(wm_candidate, iterations=1).astype(np.uint8)
            desc = "大脑深部白质 (White Matter) 提取"

        elif region == "gray_matter_cortex":
            # 大脑皮质灰质为脑表面及中等信号区域 (介于脑脊液与白质之间)
            wm_threshold = p98 * 0.65
            cortex_candidate = np.logical_and(brain_mask, smoothed < wm_threshold).astype(np.uint8)
            new_mask = ndimage.binary_opening(cortex_candidate, iterations=1).astype(np.uint8)
            desc = "大脑皮质灰质 (Gray Matter Cortex) 提取"

        else:
            new_mask = brain_mask
            desc = "全脑实质去颅骨提取"

        metrics = self.compute_metrics(old_mask, new_mask, context)
        return ToolResult(
            success=True,
            new_mask=new_mask,
            observation_metrics=metrics,
            message=f"已成功完成 {desc}。标定体素量 {metrics['new_voxel_count']} 个，解剖总体积 {metrics['current_volume_cm3']} cm³，平均组织信号强度 {metrics['mean_intensity_inside_mask']}。"
        )

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
