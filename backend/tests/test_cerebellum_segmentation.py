import os
import sys
import numpy as np
import nibabel as nib

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.tools.guided_refinement_tools import SpatialPromptGuidedSegmentationTool
from agent_core.tools.interactive_tools import DrawPolygonContourTool, ContourScissorsCutTool
from agent_core.tools.base_tool import ImageContext
from agent_core.tools.vtk_engine import VTKSegmentationEngine

def test_cerebellum_generic_segmentation():
    print("=== [Test] 验证通用空间引导分割与 VTK 剪刀算法 ===")
    
    # 检查 VTK 引擎
    print(f"VTK 算法引擎是否可用: {VTKSegmentationEngine.is_available()}")
    
    # 加载真实 MNI152 脑影像
    nii_path = os.path.join(os.path.dirname(backend_dir), "image", "MNI152NLin6_res-1x1x1_T1w.nii")
    if not os.path.exists(nii_path):
        print(f"跳过真实影像测试: {nii_path} 不存在")
        return

    nii = nib.load(nii_path)
    img_data = nii.get_fdata().astype(np.float32)
    spacing = tuple(nii.header.get_zooms()[:3])
    empty_mask = np.zeros(img_data.shape, dtype=np.uint8)
    ctx = ImageContext(img_data, empty_mask, spacing)

    # 1. 运行通用空间引导区域生长
    guided_tool = SpatialPromptGuidedSegmentationTool()
    res = guided_tool.execute(
        ctx,
        target_name="小脑",
        center_point_3d=[91, 55, 38],
        bbox_3d=[40, 20, 15, 140, 90, 60],
        tissue_intensity_type="brain_parenchyma",
        refinement_mode="replace"
    )
    
    print(f"1. 通用空间引导分割成功: {res.success}")
    metrics = res.observation_metrics
    vol_cm3 = metrics.get("current_volume_cm3", 0)
    print(f"   提取物理体积: {vol_cm3:.2f} cm³")
    assert res.success is True
    assert vol_cm3 > 10.0

    # 2. 运行 VTK 曲线剪刀裁剪
    ctx2 = ImageContext(img_data, res.new_mask, spacing)
    scissors_tool = ContourScissorsCutTool()
    scissors_poly = [
        [30, 80],
        [80, 85],
        [150, 80],
        [150, 120],
        [30, 120]
    ]
    res_cut = scissors_tool.execute(
        ctx2,
        plane="sagittal",
        polygon_points=scissors_poly,
        slice_range=[85, 95],
        cut_mode="remove_inside"
    )
    print(f"2. VTK 曲线剪刀裁切成功: {res_cut.success}")
    assert res_cut.success is True

    print("\n==========================================")
    print("🎉 通用分割与 VTK 剪刀算法验证全部通过！")
    print("==========================================")

if __name__ == "__main__":
    test_cerebellum_generic_segmentation()
