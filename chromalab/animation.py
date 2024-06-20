from .visualizer import PSWrapper, easeFunction
import numpy as np

def doNothing(j):
    pass

def SetEnabledFn(viz, list_names, list_types=None):
    if list_types is None:
        return lambda list_true: [viz.ps.get_surface_mesh(item).set_enabled(enabled) for item, enabled in zip(list_names, list_true)]
    else:
        return lambda list_true: [getattr(viz.ps, f'get_{type_name}')(item).set_enabled(enabled) for item, type_name, enabled in zip(list_names, list_types, list_true)]

def concatFns(fns):
    return lambda j: [fn(j) for fn in fns]

def FadeIn(viz, name, total_frames, opacity_range=[0, 1]):
    def fadeIn(j):
        viz.ps.get_surface_mesh(name).set_transparency(easeFunction(j/total_frames)*(opacity_range[1] - opacity_range[0])  + opacity_range[0])
    return fadeIn

def FadeOut(viz, name, total_frames, opacity_range=[0, 1], removeBefore=0):
    def fadeOut(j):
        if j == total_frames - removeBefore and removeBefore > 0:
            viz.ps.get_surface_mesh(name).set_enabled(False)
        viz.ps.get_surface_mesh(name).set_transparency(opacity_range[1]-(easeFunction(j/total_frames)*(opacity_range[1] - opacity_range[0])))
    return fadeOut

def RotateAroundZ(viz, total_frames, r, theta, phi_range, lookAt=[0, 0, 0]):
    def rotateAroundZ(j):
        phi = phi_range[0] + (phi_range[1] - phi_range[0]) * j / total_frames
        point_3d = PSWrapper.polarToCartesian(r, theta, phi)
        viz.ps.look_at(point_3d, lookAt)
    return rotateAroundZ

def RotateTheta(viz, total_frames, r, theta_range, phi, lookAt=[0, 0, 0]):
    def moveAlongTheta(j):
        theta = theta_range[0] + (theta_range[1] - theta_range[0]) * j / total_frames
        point_3d = PSWrapper.polarToCartesian(r, theta, phi)
        viz.ps.look_at(point_3d, lookAt)
    return moveAlongTheta

def MoveAlongPath(viz, total_frames, r_phi_thetas, lookAt=[0, 0, 0]): 
    r_theta_phi_interpolated = PSWrapper.getPathPoints(r_phi_thetas, total_frames, method='arc')
    def moveAlongPath(j):
        r, theta, phi = r_theta_phi_interpolated[j]
        point_3d = PSWrapper.polarToCartesian(r, theta, phi)
        viz.ps.look_at(point_3d, lookAt)
    return moveAlongPath

def CoordBasisChange(viz, total_frames):
    transforms_LMS, transforms_RGB = viz._getIntermediateTransformsLMSRGB(total_frames)
    HMat = viz._getmatrixBasisToLum()

    ps_mesh = viz.ps.get_surface_mesh("mesh")
    names = viz._getCoordBasis("RGB_coords", ((HMat@transforms_RGB[0])[:3, :3]).T, 1)
    names += viz._getCoordBasis("LMS_coords", ((HMat@transforms_LMS[0])[:3, :3]).T, 1)

    def expandBasis(j):
        # [viz.ps.get_surface_mesh(n).remove() for n in names]
        ps_mesh.set_transform(HMat@transforms_LMS[j])
        viz._getCoordBasis("RGB_coords", ((HMat@transforms_RGB[j])[:3, :3]).T)
        viz._getCoordBasis("LMS_coords", ((HMat@transforms_LMS[j])[:3, :3]).T)
    
    return expandBasis, HMat@transforms_LMS[-1]
    
