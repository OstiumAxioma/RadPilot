import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.tools.morphology_tools import (
    AnalyzeConnectivityTool,
    ConnectedComponentFilterTool
)
from agent_core.tools.base_tool import ImageContext
from agent_core.tools.registry import GLOBAL_TOOL_REGISTRY

def test_connectivity_analysis_and_filtering():
    print("=== [Test] 测试主动连通性拓扑分析探针与多维度孤岛精准过滤 ===")
    
    shape = (100, 100, 100)
    spacing = (1.0, 1.0, 1.0)
    fake_img = np.zeros(shape, dtype=np.float32)
    fake_mask = np.zeros(shape, dtype=np.uint8)
    
    # 构造 3 个不同大小且互不连通的解剖区域:
    # 1. 主器官 (球体, 半径 15, 中心 [50, 50, 50])
    x, y, z = np.ogrid[:100, :100, :100]
    main_obj = ((x - 50)**2 + (y - 50)**2 + (z - 50)**2) <= 15**2
    fake_mask[main_obj] = 1
    
    # 2. 次级粘连结构 (球体, 半径 6, 中心 [80, 80, 80])
    sub_obj = ((x - 80)**2 + (y - 80)**2 + (z - 80)**2) <= 6**2
    fake_mask[sub_obj] = 1

    # 3. 微小碎屑伪影 (球体, 半径 2, 中心 [20, 20, 20])
    noise_obj = ((x - 20)**2 + (y - 20)**2 + (z - 20)**2) <= 2**2
    fake_mask[noise_obj] = 1

    ctx = ImageContext(fake_img, fake_mask, spacing)

    # 1. 运行主动连通性扫描探针 (AnalyzeConnectivityTool)
    conn_tool = GLOBAL_TOOL_REGISTRY.get_tool("analyze_connectivity")
    assert conn_tool is not None, "未找到 analyze_connectivity 工具"

    res_diag = conn_tool.execute(ctx)
    print(f"1. 连通性扫描成功: {res_diag.success}")
    metrics = res_diag.observation_metrics
    total_islands = metrics.get("total_islands", 0)
    islands_list = metrics.get("islands_analyzed", [])
    
    print(f"   检测到独立连通域总数: {total_islands} 个")
    assert total_islands == 3
    assert len(islands_list) == 3
    
    # 验证最大主体档案
    main_island = islands_list[0]
    print(f"   Island 1 (主器官): 体积={main_island['volume_cm3']} cm³, 质心={main_island['centroid_3d']}, 占比={main_island['volume_ratio_pct']}%")
    assert main_island["volume_cm3"] > 10.0
    assert np.allclose(main_island["centroid_3d"], [50, 50, 50], atol=1.0)

    # 2. 验证多维度过滤工具 (ConnectedComponentFilterTool)
    filter_tool = GLOBAL_TOOL_REGISTRY.get_tool("filter_connected_components")
    assert filter_tool is not None, "未找到 filter_connected_components 工具"

    # 测试点包含模式: 仅保留包覆 [80, 80, 80] 的 Island 2
    res_point_filter = filter_tool.execute(ctx, keep_point_3d=[80, 80, 80])
    print(f"2. 点包含精准过滤: 成功={res_point_filter.success}, 提取体积={res_point_filter.observation_metrics.get('current_volume_cm3')} cm³")
    assert res_point_filter.success is True
    assert res_point_filter.new_mask[80, 80, 80] == 1
    assert res_point_filter.new_mask[50, 50, 50] == 0

    # 测试 Top-1 模式: 仅保留最大主体
    res_top1 = filter_tool.execute(ctx, keep_top_k=1)
    print(f"3. Top-1 最大主体保留: 成功={res_top1.success}, 体积={res_top1.observation_metrics.get('current_volume_cm3')} cm³")
    assert res_top1.success is True
    assert res_top1.new_mask[50, 50, 50] == 1
    assert res_top1.new_mask[80, 80, 80] == 0
    assert res_top1.new_mask[20, 20, 20] == 0

    print("\n==========================================")
    print("🎉 连通性深度拓扑分析探针与多维过滤测试全部通过！")
    print("==========================================")

if __name__ == "__main__":
    test_connectivity_analysis_and_filtering()
