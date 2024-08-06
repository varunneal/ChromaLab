from .visualizer import PSWrapper, easeFunction
import numpy as np


class Animation:
    def __init__(self, viz, dict_name_types) -> None:
        self.viz = viz
        self.dict = dict_name_types

    @staticmethod
    def doNothing(j):
        pass

    def SetEnabledFn(self):
        return lambda list_true: [getattr(self.viz.ps, f'get_{type_name}')(item).set_enabled(enabled) for item, type_name, enabled in zip(list(self.dict.keys()), list(self.dict.values()), list_true)]

    def set_enabled(self, name, toggle):
        return getattr(self.viz.ps, f'get_{self.dict[name]}')(name).set_enabled(toggle)
    
    def set_transparency(self, name, value):
        return getattr(self.viz.ps, f'get_{self.dict[name]}')(name).set_transparency(value)

    def ResetTransparencies(self):
        [getattr(self.viz.ps, f'get_{type_name}')(item).set_transparency(1) for item, type_name in zip(list(self.dict.keys()), list(self.dict.values()))]

    @staticmethod
    def concatFns(fns):
        return lambda j: [fn(j) for fn in fns]

    def FadeIn(self, name, total_frames, opacity_range=[0, 1]):
        def fadeIn(j):
            getattr(self.viz.ps, f'get_{self.dict[name]}')(name).set_transparency(easeFunction(j/total_frames)*(opacity_range[1] - opacity_range[0])  + opacity_range[0])
        return fadeIn

    def FadeOut(self, name, total_frames, opacity_range=[0, 1], removeBefore=0):
        def fadeOut(j):
            if j == total_frames - removeBefore and removeBefore > 0:
                getattr(self.viz.ps, f'get_{self.dict[name]}')(name).set_enabled(False)
            getattr(self.viz.ps, f'get_{self.dict[name]}')(name).set_transparency(opacity_range[1]-(easeFunction(j/total_frames)*(opacity_range[1] - opacity_range[0])))
        return fadeOut

    def RotateAroundZ(self, total_frames, r, theta, phi_range, lookAt=[0, 0, 0]):
        if phi_range[0] == phi_range[1]:
            phi_range[1] = 360
        def rotateAroundZ(j):
            phi = phi_range[0] + (phi_range[1] - phi_range[0]) * j / total_frames
            point_3d = PSWrapper.polarToCartesian(r, theta, phi)
            self.viz.ps.look_at(point_3d, lookAt)
        return rotateAroundZ, phi_range[1] % 360

    def RotateTheta(self, total_frames, r, theta_range, phi, lookAt=[0, 0, 0]):
        def moveAlongTheta(j):
            theta = theta_range[0] + (theta_range[1] - theta_range[0]) * j / total_frames
            point_3d = PSWrapper.polarToCartesian(r, theta, phi)
            self.viz.ps.look_at(point_3d, lookAt)
        return moveAlongTheta

    def MoveAlongPath(self, total_frames, r_phi_thetas, lookAt=[0, 0, 0]): 
        r_theta_phi_interpolated = PSWrapper.getPathPoints(r_phi_thetas, total_frames, method='arc')
        def moveAlongPath(j):
            r, theta, phi = r_theta_phi_interpolated[j]
            point_3d = PSWrapper.polarToCartesian(r, theta, phi)
            self.viz.ps.look_at(point_3d, lookAt)
        return moveAlongPath

    def CoordBasisChange(self,  total_frames):
        transforms_LMS, transforms_RGB = self.viz._getIntermediateTransformsLMSRGB(total_frames)
        HMat = self.viz._getmatrixBasisToLum()

        ps_mesh = self.viz.ps.get_surface_mesh("mesh")
        names = self.viz._getCoordBasis("RGB_coords", ((HMat@transforms_RGB[0])[:3, :3]).T, 1)
        names += self.viz._getCoordBasis("LMS_coords", ((HMat@transforms_LMS[0])[:3, :3]).T, 1)

        def expandBasis(j):
            # [self.viz.ps.get_surface_mesh(n).remove() for n in names]
            ps_mesh.set_transform(HMat@transforms_LMS[j])
            self.viz._getCoordBasis("RGB_coords", ((HMat@transforms_RGB[j])[:3, :3]).T)
            self.viz._getCoordBasis("LMS_coords", ((HMat@transforms_LMS[j])[:3, :3]).T)
        
        return expandBasis, HMat@transforms_LMS[-1]
        
