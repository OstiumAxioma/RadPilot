from typing import Dict, List, Any, Optional
from .base_tool import BaseMedicalTool, ImageContext, ToolResult
from .interactive_tools import (
    ThresholdRangeTool,
    PaintBrush3DTool,
    EraseBrush3DTool,
    ScissorsCutTool,
    RegionGrowthTool,
    FillBetweenSlicesTool,
    IslandAndSmoothTool,
    DrawPolygonContourTool,
    ContourScissorsCutTool,
    MarkerControlledWatershedTool,
    SmartIntensityBrushTool
)
from .inspection_tools import (
    InspectOrthoSliceTool,
    BrowseSliceGalleryTool
)
from .guided_refinement_tools import SpatialPromptGuidedSegmentationTool
from .segmentation_tools import (
    IntensityThresholdSegmentationTool,
    ResetMaskTool
)
from .morphology_tools import (
    AnalyzeConnectivityTool,
    MorphologicalDilationTool,
    MorphologicalErosionTool,
    ConnectedComponentFilterTool,
    MorphologicalSmoothTool
)

class ToolRegistry:
    """
    医学图像工具集中注册表
    负责工具的依赖注入、参数校验、Schema 导出与统一调度
    """
    def __init__(self):
        self._tools: Dict[str, BaseMedicalTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册所有内置核心医学图像工具"""
        default_tools = [
            # 1. 放射学主动视觉探针与切片画廊导航
            InspectOrthoSliceTool(),
            BrowseSliceGalleryTool(),
            # 2. 主动连通性深度拓扑检测与诊断探针
            AnalyzeConnectivityTool(),
            # 3. 连续曲线多边形绘制与自由剪刀
            DrawPolygonContourTool(),
            ContourScissorsCutTool(),
            # 4. 原生多模态空间引导与区域生长
            SpatialPromptGuidedSegmentationTool(),
            # 5. 高级图割与 PS 智能边缘吸附工具箱
            MarkerControlledWatershedTool(),
            SmartIntensityBrushTool(),
            # 6. 放射科通用精细原子工具箱 (3D Slicer / ITK-SNAP 范式)
            ThresholdRangeTool(),
            RegionGrowthTool(),
            PaintBrush3DTool(),
            EraseBrush3DTool(),
            ScissorsCutTool(),
            FillBetweenSlicesTool(),
            IslandAndSmoothTool(),
            # 7. 通用形态学与连通域过滤算子
            ConnectedComponentFilterTool(),
            MorphologicalDilationTool(),
            MorphologicalErosionTool(),
            ConnectedComponentFilterTool(),
            MorphologicalSmoothTool(),
            IntensityThresholdSegmentationTool(),
            ResetMaskTool()
        ]
        for t in default_tools:
            self.register(t)

    def register(self, tool: BaseMedicalTool):
        """注册单个医学工具"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseMedicalTool]:
        """按名称查找工具"""
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseMedicalTool]:
        """获取所有可用工具列表"""
        return list(self._tools.values())

    def get_function_declarations(self) -> List[Dict[str, Any]]:
        """
        导出所有工具的 Gemini Function Calling 标准声明数组
        """
        return [tool.to_function_declaration() for tool in self._tools.values()]

    def execute_tool(self, name: str, context: ImageContext, **kwargs) -> ToolResult:
        """
        统一调度执行工具，具备安全边界与异常拦截
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                new_mask=context.current_mask,
                observation_metrics={},
                message=f"未找到名为 '{name}' 的医学图像算子。",
                error_message=f"TOOL_NOT_FOUND: {name}"
            )
        try:
            return tool.execute(context, **kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                new_mask=context.current_mask,
                observation_metrics={},
                message=f"执行算子 '{name}' 时发生异常: {str(e)}",
                error_message=str(e)
            )

# 全局单例
GLOBAL_TOOL_REGISTRY = ToolRegistry()
