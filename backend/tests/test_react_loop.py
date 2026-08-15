import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.tools.interactive_tools import (
    ThresholdRangeTool,
    PaintBrush3DTool,
    EraseBrush3DTool,
    ScissorsCutTool,
    RegionGrowthTool,
    FillBetweenSlicesTool,
    IslandAndSmoothTool
)
from agent_core.tools.base_tool import ImageContext

def test_interactive_tools():
    print("=== 测试 7 大放射科原子工具 ===")
    
    shape = (100, 100, 100)
    spacing = (1.0, 1.0, 1.0)
    fake_img = np.zeros(shape, dtype=np.float32)
    fake_img[30:70, 30:70, 30:70] = 5000.0  # 中心高信号块
    
    empty_mask = np.zeros(shape, dtype=np.uint8)
    ctx = ImageContext(fake_img, empty_mask, spacing)

    # 1. threshold_range
    t_tool = ThresholdRangeTool()
    res1 = t_tool.execute(ctx, min_intensity=4000, max_intensity=6000)
    print(f"1. ThresholdRange: 成功={res1.success}, 体积={res1.observation_metrics.get('current_volume_cm3')} cm³")
    assert res1.observation_metrics.get('current_volume_cm3') > 0

    # 2. paint_brush_3d
    ctx.current_mask = res1.new_mask
    p_tool = PaintBrush3DTool()
    res2 = p_tool.execute(ctx, center=[25, 25, 25], radius_mm=5.0, mode="add")
    print(f"2. PaintBrush3D: 成功={res2.success}, 体积增量={res2.observation_metrics.get('volume_change_mm3')} mm³")
    assert res2.observation_metrics.get('volume_change_mm3') > 0

    # 3. erase_brush_3d
    ctx.current_mask = res2.new_mask
    e_tool = EraseBrush3DTool()
    res3 = e_tool.execute(ctx, center=[25, 25, 25], radius_mm=6.0)
    print(f"3. EraseBrush3D: 成功={res3.success}, 体积增量={res3.observation_metrics.get('volume_change_mm3')} mm³")
    assert res3.observation_metrics.get('volume_change_mm3') < 0

    # 4. scissors_cut
    ctx.current_mask = res3.new_mask
    s_tool = ScissorsCutTool()
    res4 = s_tool.execute(ctx, plane="axial", cut_index=50, remove_side="greater_than")
    print(f"4. ScissorsCut: 成功={res4.success}, 裁切后体积={res4.observation_metrics.get('current_volume_cm3')} cm³")

    # 5. region_growth
    ctx.current_mask = empty_mask
    rg_tool = RegionGrowthTool()
    res5 = rg_tool.execute(ctx, seed_point=[50, 50, 40], intensity_tolerance=1000)
    print(f"5. RegionGrowth: 成功={res5.success}, 生长体积={res5.observation_metrics.get('current_volume_cm3')} cm³")

    # 6. fill_between_slices
    ctx.current_mask = np.zeros(shape, dtype=np.uint8)
    ctx.current_mask[40:60, 40:60, 20] = 1
    ctx.current_mask[40:60, 40:60, 30] = 1
    f_tool = FillBetweenSlicesTool()
    res6 = f_tool.execute(ctx, axis="axial", slice_start=20, slice_end=30)
    print(f"6. FillBetweenSlices: 成功={res6.success}, 插值填充体积={res6.observation_metrics.get('volume_change_mm3')} mm³")
    assert res6.observation_metrics.get('volume_change_mm3') > 0

    # 7. island_and_smooth
    ctx.current_mask = res6.new_mask
    ctx.current_mask[5, 5, 5] = 1
    is_tool = IslandAndSmoothTool()
    res7 = is_tool.execute(ctx, min_volume_mm3=5.0, fill_holes=True)
    print(f"7. IslandAndSmooth: 成功={res7.success}, 孤岛过滤后体积={res7.observation_metrics.get('current_volume_cm3')} cm³")

    print("\n==========================================")
    print("🎉 backend/tests/test_react_loop.py 全部通过！")
    print("==========================================")

if __name__ == "__main__":
    test_interactive_tools()
