import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.tools.interactive_tools import (
    DrawPolygonContourTool,
    ContourScissorsCutTool,
    PaintBrush3DTool,
    EraseBrush3DTool
)
from agent_core.tools.base_tool import ImageContext
from agent_core.tools.registry import GLOBAL_TOOL_REGISTRY

def test_polygon_and_dual_mode_brushes():
    print("=== [Test] 测试多边形剪刀与切片/3D双模态画笔/橡皮擦工具 ===")
    
    shape = (100, 100, 100)
    spacing = (1.0, 1.0, 1.0)
    fake_img = np.zeros(shape, dtype=np.float32)
    empty_mask = np.zeros(shape, dtype=np.uint8)
    ctx = ImageContext(fake_img, empty_mask, spacing)

    # 1. 在正中矢状位 X=50 绘制一个有机三角多边形
    draw_tool = GLOBAL_TOOL_REGISTRY.get_tool("draw_polygon_contour")
    poly_points = [
        [30, 20],
        [70, 25],
        [80, 60],
        [50, 75],
        [35, 50]
    ]
    res1 = draw_tool.execute(ctx, plane="sagittal", slice_index=50, points=poly_points, thickness_voxels=3, mode="add")
    print(f"1. DrawPolygonContour: 成功={res1.success}, 生成体积={res1.observation_metrics.get('current_volume_cm3')} cm³")
    assert res1.success is True
    assert res1.observation_metrics.get("current_volume_cm3") > 0

    # 2. 2D 切片模式精准画笔涂抹 (在矢状面 X=50, Y=40, Z=30 点涂)
    ctx.current_mask = res1.new_mask
    paint_tool = GLOBAL_TOOL_REGISTRY.get_tool("paint_brush_3d")
    res_paint = paint_tool.execute(ctx, plane="sagittal", slice_index=50, point_2d=[40, 30], radius_mm=4.0)
    print(f"2. 2D切片画笔涂抹: 成功={res_paint.success}, 涂抹后体积={res_paint.observation_metrics.get('current_volume_cm3')} cm³")
    assert res_paint.success is True

    # 3. 2D 切片模式精准橡皮擦擦除局部突刺 (在轴位 Z=60, X=50, Y=80 擦除)
    ctx.current_mask = res_paint.new_mask
    erase_tool = GLOBAL_TOOL_REGISTRY.get_tool("erase_brush_3d")
    res_erase = erase_tool.execute(ctx, plane="axial", slice_index=60, point_2d=[50, 80], radius_mm=6.0)
    print(f"3. 2D切片橡皮擦局部擦除: 成功={res_erase.success}, 擦除后体积={res_erase.observation_metrics.get('current_volume_cm3')} cm³")
    assert res_erase.success is True
    assert res_erase.observation_metrics.get("volume_change_mm3") < 0

    # 4. 使用连续曲线多边形剪刀裁切
    ctx.current_mask = res_erase.new_mask
    cut_tool = GLOBAL_TOOL_REGISTRY.get_tool("contour_scissors_cut")
    cut_points = [
        [40, 55],
        [85, 55],
        [85, 80],
        [40, 80]
    ]
    res_cut = cut_tool.execute(ctx, plane="sagittal", polygon_points=cut_points, cut_mode="remove_inside")
    print(f"4. ContourScissorsCut: 成功={res_cut.success}, 裁切后体积={res_cut.observation_metrics.get('current_volume_cm3')} cm³")
    assert res_cut.success is True

    print("\n==========================================")
    print("🎉 多模态画笔/橡皮擦与曲线剪刀测试全部通过！")
    print("==========================================")

if __name__ == "__main__":
    test_polygon_and_dual_mode_brushes()
