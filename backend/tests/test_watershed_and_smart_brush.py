import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.tools.interactive_tools import (
    MarkerControlledWatershedTool,
    SmartIntensityBrushTool
)
from agent_core.tools.base_tool import ImageContext
from agent_core.tools.registry import GLOBAL_TOOL_REGISTRY

def test_watershed_and_smart_brush():
    print("=== [Test] 测试分水岭解剖图割 (Watershed) 与 PS 智能吸附画笔 ===")
    
    # 构造带边缘对比度的人工三维体数据
    shape = (100, 100, 100)
    spacing = (1.0, 1.0, 1.0)
    fake_img = np.ones(shape, dtype=np.float32) * 500.0  # 背景低强度 (如脑脊液)
    
    # 中心放入一个高强度的椭球体 (如器官组织，强度 1500)
    x, y, z = np.ogrid[:100, :100, :100]
    ellipsoid = (((x - 50)**2 / 20**2) + ((y - 50)**2 / 20**2) + ((z - 50)**2 / 20**2)) <= 1.0
    fake_img[ellipsoid] = 1500.0
    
    empty_mask = np.zeros(shape, dtype=np.uint8)
    ctx = ImageContext(fake_img, empty_mask, spacing)

    # 1. 测试分水岭图割算子 (Watershed)
    ws_tool = GLOBAL_TOOL_REGISTRY.get_tool("watershed_segmentation")
    assert ws_tool is not None, "未找到 watershed_segmentation 工具"

    # 提供中心前景点与外围背景点
    res_ws = ws_tool.execute(
        ctx,
        foreground_points=[[50, 50, 50], [52, 48, 50]],
        background_points=[[10, 10, 10], [90, 90, 90], [50, 50, 80]],
        compactness=0.05,
        mode="replace"
    )
    print(f"1. 分水岭分割: 成功={res_ws.success}, 提取体积={res_ws.observation_metrics.get('current_volume_cm3')} cm³")
    assert res_ws.success is True
    assert res_ws.observation_metrics.get("current_volume_cm3") > 10.0

    # 2. 测试 2D 切片分水岭 (在轴位 Z=50 上传入二维前背景点)
    res_ws_2d = ws_tool.execute(
        ctx,
        plane="axial",
        slice_index=50,
        fg_points_2d=[[50, 50]],
        bg_points_2d=[[10, 10], [90, 90]],
        mode="replace"
    )
    print(f"2. 2D切片分水岭: 成功={res_ws_2d.success}, 体积={res_ws_2d.observation_metrics.get('current_volume_cm3')} cm³")
    assert res_ws_2d.success is True

    # 3. 测试 PS 智能边缘吸附画笔 (Smart Intensity Brush)
    smart_tool = GLOBAL_TOOL_REGISTRY.get_tool("smart_intensity_brush")
    assert smart_tool is not None, "未找到 smart_intensity_brush 工具"

    ctx.current_mask = np.zeros(shape, dtype=np.uint8)
    res_smart = smart_tool.execute(
        ctx,
        plane="sagittal",
        slice_index=50,
        point_2d=[50, 50],
        radius_mm=25.0,  # 即使半径很大，也应该被椭球体与背景的高梯度边界自动阻断！
        mode="add"
    )
    print(f"3. PS智能吸附画笔: 成功={res_smart.success}, 提取体积={res_smart.observation_metrics.get('current_volume_cm3')} cm³")
    assert res_smart.success is True
    assert res_smart.observation_metrics.get("current_volume_cm3") > 0.0

    print("\n==========================================")
    print("🎉 分水岭与 PS 智能吸附画笔测试全部通过！")
    print("==========================================")

if __name__ == "__main__":
    test_watershed_and_smart_brush()
