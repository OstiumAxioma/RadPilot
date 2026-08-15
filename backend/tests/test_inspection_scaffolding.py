import os
import sys
import numpy as np
import nibabel as nib

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.tools.inspection_tools import InspectOrthoSliceTool, BrowseSliceGalleryTool
from agent_core.tools.base_tool import ImageContext

def test_inspection_scaffolding():
    print("=== [Test] 验证主动视觉探针与切片导航脚手架工具 ===")
    
    nii_path = os.path.join(os.path.dirname(backend_dir), "image", "MNI152NLin6_res-1x1x1_T1w.nii")
    if not os.path.exists(nii_path):
        print(f"跳过真实影像测试: {nii_path} 不存在")
        return

    nii = nib.load(nii_path)
    img_data = nii.get_fdata().astype(np.float32)
    spacing = tuple(nii.header.get_zooms()[:3])
    
    # 模拟一个带简单掩码的上下文
    fake_mask = np.zeros(img_data.shape, dtype=np.uint8)
    fake_mask[80:100, 40:70, 20:50] = 1
    ctx = ImageContext(img_data, fake_mask, spacing)

    # 1. 测试主动调取矢状位正中切片 X=91 (带变焦)
    inspect_tool = InspectOrthoSliceTool()
    res1 = inspect_tool.execute(ctx, plane="sagittal", slice_index=91, zoom_roi=[20, 20, 120, 120], overlay_mask=True)
    print(f"1. inspect_ortho_slice: 成功={res1.success}, 是否附带高分辨率图像部件={res1.attached_image_part is not None}")
    assert res1.success is True
    assert res1.attached_image_part is not None
    assert "inlineData" in res1.attached_image_part

    # 2. 测试浏览后颅窝小脑连续切片画廊 Z=20~50
    gallery_tool = BrowseSliceGalleryTool()
    res2 = gallery_tool.execute(ctx, plane="axial", start_slice=20, end_slice=50, step=6)
    print(f"2. browse_slice_gallery: 成功={res2.success}, 画廊层数={len(res2.observation_metrics.get('browsed_slices', []))}")
    assert res2.success is True
    assert res2.attached_image_part is not None

    print("\n==========================================")
    print("🎉 主动切片探针与连续画廊脚手架工具验证全部通过！")
    print("==========================================")

if __name__ == "__main__":
    test_inspection_scaffolding()
