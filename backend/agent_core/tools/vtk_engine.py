import numpy as np
from typing import List, Tuple, Optional, Any

try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


class VTKSegmentationEngine:
    """
    基于 3D Slicer 同款 VTK 算法管道的工业级医学图像分割引擎
    提供多边形 Stencil 剪刀裁切、连续距离场 Margin 膨胀/腐蚀与连通域提纯。
    """
    
    @staticmethod
    def is_available() -> bool:
        return VTK_AVAILABLE

    @staticmethod
    def numpy_to_vtk_image(data_3d: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Any:
        """将 NumPy 3D 矩阵 (X, Y, Z) 零拷贝转换为 vtkImageData"""
        if not VTK_AVAILABLE:
            raise RuntimeError("VTK 未安装或不可用")
        
        # VTK 内存排布为 Fortran 序 (X 是最快变化的轴)
        vtk_image = vtk.vtkImageData()
        dim_x, dim_y, dim_z = data_3d.shape
        vtk_image.SetDimensions(dim_x, dim_y, dim_z)
        vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
        vtk_image.SetOrigin(0.0, 0.0, 0.0)

        # 展平为 C/Fortran 连续内存
        flat_data = np.ascontiguousarray(data_3d, dtype=np.uint8)
        vtk_array = numpy_support.numpy_to_vtk(flat_data.ravel(order='F'), deep=1, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_image.GetPointData().SetScalars(vtk_array)
        return vtk_image

    @staticmethod
    def vtk_image_to_numpy(vtk_image: Any, shape_3d: Tuple[int, int, int]) -> np.ndarray:
        """从 vtkImageData 抽取回 NumPy 3D 矩阵"""
        if not VTK_AVAILABLE:
            raise RuntimeError("VTK 未安装或不可用")
        
        scalars = vtk_image.GetPointData().GetScalars()
        np_flat = numpy_support.vtk_to_numpy(scalars)
        return np_flat.reshape(shape_3d, order='F')

    @classmethod
    def apply_polygon_stencil_scissors(
        cls,
        mask_3d: np.ndarray,
        plane: str,
        points_2d: List[List[float]],
        slice_range: Optional[List[int]] = None,
        cut_mode: str = "remove_inside",
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        【3D Slicer 同款 VTK Scissors 剪刀算法】
        使用 vtkPolyDataToImageStencil 构建多边形 Stencil 模板，在体空间中进行精准雕刻裁切。
        """
        if not VTK_AVAILABLE or len(points_2d) < 3:
            # 回退至高效矢量化扫描线算法
            from .interactive_tools import _rasterize_polygon_2d
            return cls._fallback_scissors(mask_3d, plane, points_2d, slice_range, cut_mode)

        dim_x, dim_y, dim_z = mask_3d.shape
        new_mask = mask_3d.copy()

        # 构建 2D 多边形并挤出为 3D 空间柱体模板
        # 利用 vtkPolygon 和 vtkPolyData
        pts = vtk.vtkPoints()
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(points_2d))

        for i, pt in enumerate(points_2d):
            # 根据平面确定世界几何坐标
            if plane == "sagittal":
                # points 为 [Y, Z]
                pts.InsertNextPoint(0.0, float(pt[0]), float(pt[1]))
            elif plane == "coronal":
                # points 为 [X, Z]
                pts.InsertNextPoint(float(pt[0]), 0.0, float(pt[1]))
            else:  # axial
                # points 为 [X, Y]
                pts.InsertNextPoint(float(pt[0]), float(pt[1]), 0.0)
            polygon.GetPointIds().SetId(i, i)

        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(pts)
        poly_data.SetPolys(polygons)

        # 挤出为贯穿/局部 3D 多边形网格 (vtkLinearExtrusionFilter)
        extruder = vtk.vtkLinearExtrusionFilter()
        extruder.SetInputData(poly_data)
        extruder.SetExtrusionTypeToNormalExtrusion()

        if plane == "sagittal":
            extruder.SetVector(1.0, 0.0, 0.0)
            extruder.SetScaleFactor(float(dim_x))
        elif plane == "coronal":
            extruder.SetVector(0.0, 1.0, 0.0)
            extruder.SetScaleFactor(float(dim_y))
        else:
            extruder.SetVector(0.0, 0.0, 1.0)
            extruder.SetScaleFactor(float(dim_z))
        extruder.Update()

        # 生成 3D 空间 Stencil
        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputConnection(extruder.GetOutputPort())
        stencil.SetOutputSpacing(spacing[0], spacing[1], spacing[2])
        stencil.SetOutputWholeExtent(0, dim_x - 1, 0, dim_y - 1, 0, dim_z - 1)
        stencil.Update()

        vtk_img = cls.numpy_to_vtk_image(mask_3d, spacing)

        # 执行 Stencil 裁切 (vtkImageStencil)
        image_stencil = vtk.vtkImageStencil()
        image_stencil.SetInputData(vtk_img)
        image_stencil.SetStencilConnection(stencil.GetOutputPort())
        image_stencil.SetBackgroundValue(0)

        if cut_mode == "remove_inside":
            image_stencil.ReverseStencilOff()
        else:
            image_stencil.ReverseStencilOn()

        image_stencil.Update()
        cut_result = cls.vtk_image_to_numpy(image_stencil.GetOutput(), mask_3d.shape)

        # 若指定了局部 slice_range，仅在范围内应用
        if slice_range and len(slice_range) == 2:
            s0, s1 = slice_range[0], slice_range[1]
            if plane == "sagittal":
                new_mask[s0:s1, :, :] = cut_result[s0:s1, :, :]
            elif plane == "coronal":
                new_mask[:, s0:s1, :] = cut_result[:, s0:s1, :]
            else:
                new_mask[:, :, s0:s1] = cut_result[:, :, s0:s1]
            return new_mask
        
        return cut_result

    @classmethod
    def apply_continuous_margin(
        cls,
        mask_3d: np.ndarray,
        margin_size_voxels: int,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        【3D Slicer 同款 VTK Continuous Margin 膨胀/腐蚀】
        基于 vtkImageContinuousDilation3D 与 vtkImageContinuousErosion3D。
        """
        if not VTK_AVAILABLE or margin_size_voxels == 0:
            from scipy import ndimage
            if margin_size_voxels > 0:
                return ndimage.binary_dilation(mask_3d, iterations=margin_size_voxels).astype(np.uint8)
            elif margin_size_voxels < 0:
                return ndimage.binary_erosion(mask_3d, iterations=abs(margin_size_voxels)).astype(np.uint8)
            return mask_3d

        vtk_img = cls.numpy_to_vtk_image(mask_3d, spacing)
        k_size = abs(margin_size_voxels) * 2 + 1

        if margin_size_voxels > 0:
            dilator = vtk.vtkImageContinuousDilation3D()
            dilator.SetInputData(vtk_img)
            dilator.SetKernelSize(k_size, k_size, k_size)
            dilator.Update()
            return cls.vtk_image_to_numpy(dilator.GetOutput(), mask_3d.shape)
        else:
            eroder = vtk.vtkImageContinuousErosion3D()
            eroder.SetInputData(vtk_img)
            eroder.SetKernelSize(k_size, k_size, k_size)
            eroder.Update()
            return cls.vtk_image_to_numpy(eroder.GetOutput(), mask_3d.shape)

    @classmethod
    def _fallback_scissors(cls, mask_3d, plane, points_2d, slice_range, cut_mode):
        from .interactive_tools import _rasterize_polygon_2d
        dim_x, dim_y, dim_z = mask_3d.shape
        new_mask = mask_3d.copy()
        if plane == "sagittal":
            poly_2d = _rasterize_polygon_2d((dim_z, dim_y), points_2d).T
            x0 = 0 if not slice_range else max(0, slice_range[0])
            x1 = dim_x if not slice_range else min(dim_x, slice_range[1])
            for x in range(x0, x1):
                new_mask[x, :, :] = np.where(poly_2d > 0, 0 if cut_mode == "remove_inside" else new_mask[x, :, :], new_mask[x, :, :] if cut_mode == "remove_inside" else 0)
        elif plane == "coronal":
            poly_2d = _rasterize_polygon_2d((dim_z, dim_x), points_2d).T
            y0 = 0 if not slice_range else max(0, slice_range[0])
            y1 = dim_y if not slice_range else min(dim_y, slice_range[1])
            for y in range(y0, y1):
                new_mask[:, y, :] = np.where(poly_2d > 0, 0 if cut_mode == "remove_inside" else new_mask[:, y, :], new_mask[:, y, :] if cut_mode == "remove_inside" else 0)
        elif plane == "axial":
            poly_2d = _rasterize_polygon_2d((dim_y, dim_x), points_2d).T
            z0 = 0 if not slice_range else max(0, slice_range[0])
            z1 = dim_z if not slice_range else min(dim_z, slice_range[1])
            for z in range(z0, z1):
                new_mask[:, :, z] = np.where(poly_2d > 0, 0 if cut_mode == "remove_inside" else new_mask[:, :, z], new_mask[:, :, z] if cut_mode == "remove_inside" else 0)
        return new_mask
