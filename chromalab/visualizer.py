from tqdm import tqdm
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import glm
import polyscope as ps
import subprocess
import os
import pickle
import alphashape
import cv2

from .observer import Observer, transformToChromaticity, getHeringMatrix
from .spectra import Spectra
from .maxbasis import MaxBasis

def getCylinderTransform(endpoints):
    a = endpoints[1]-endpoints[0]
    a = glm.vec3(a[0], a[1], a[2])
    b = glm.vec3(0, 0, 1)
    
    mat = glm.mat4()
    # translate
    mat = glm.translate(mat, glm.vec3(endpoints[0][0], endpoints[0][1], endpoints[0][2]))
    # rotate
    v = glm.cross(b, a)
    if v!= glm.vec3(0, 0, 0):
        angle = glm.acos(glm.dot(b, a) / (glm.length(b) * glm.length(a)))
        mat = glm.rotate(mat, angle, v)

    # scale
    scale_factor = glm.length(a)
    mat = glm.scale(mat, glm.vec3(scale_factor, scale_factor, scale_factor))

    return mat

def exportDirectoryToVideo(dir_name)->None:
    # Get all image file names from the directory
    image_files = [f for f in os.listdir(dir_name) if f.startswith("frame_") and f.endswith(".png")]

    # Sort the image file names in ascending order
    image_files.sort()

    # Read each image and process it
    for image_file in image_files:
        img = cv2.imread(os.path.join(dir_name, image_file))

        # Center crop the image to be square and centered
        img_height, img_width, _ = img.shape
        crop_size = min(img_height, img_width)
        start_x = (img_width - crop_size) // 2
        start_y = (img_height - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size
        cropped_img = img[start_y:end_y, start_x:end_x]

        # Save the cropped image
        cv2.imwrite(f"{dir_name}/cropped_{image_file}", cropped_img)
    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", f"{dir_name}/cropped_frame_%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{dir_name}.mp4"])
    return

def playVideo(video_filepath):
    cap = cv2.VideoCapture(video_filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def exportAndPlay(dirname):
    exportDirectoryToVideo(dirname)
    playVideo(dirname + ".mp4")


def easeFunction(t):
    if t < 0: return 0
    if t > 1: return 1
    return -2 * t **3 + 3 * t ** 2


class DisplayBasisType(Enum):
    CONE = 1
    MAXBASIS = 2
    HERING = 3

class GeometryPrimitives:

    def __init__(self) -> None:
        self.objects = []

    def add_obj(self, obj:o3d.geometry.TriangleMesh)->None:
        self.objects.append(obj)
    
    @staticmethod
    def collapseMeshObjects(objects):
        mesh = o3d.geometry.TriangleMesh()
        for obj in objects:
            mesh += obj
        return mesh

    @staticmethod
    def createSphere(radius=0.025, center=[0, 0, 0], resolution=20, color=[0, 0, 0])->o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        mesh.translate(center)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([color]*len(mesh.vertices)))
        mesh.compute_vertex_normals()
        return mesh
    
    @staticmethod
    def createCylinder(endpoints, radius=0.025/2, resolution=20, color=[0, 0, 0])->o3d.geometry.TriangleMesh:
        #canonical cylinder is along z axis with height 1 and centered
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=1, resolution=resolution)
        mesh.translate([0, 0, 1/2])

        matrix = getCylinderTransform(endpoints)
        mesh.transform(matrix)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([color]*len(mesh.vertices)))
        mesh.compute_vertex_normals()
        return mesh
    
    @staticmethod
    def calculate_zy_rotation_for_arrow(vec):
        gamma = np.arctan2(vec[1], vec[0])
        Rz = np.array([
                        [np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]
                    ])

        vec = Rz.T @ vec

        beta = np.arctan2(vec[0], vec[2])
        Ry = np.array([
                        [np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [-np.sin(beta), 0, np.cos(beta)]
                    ])
        return Rz, Ry
    
    @staticmethod
    def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):
        assert(not np.all(end == origin))
        vec = end - origin
        size = np.sqrt(np.sum(vec**2))
        ratio_cone_cylinder = 0.15
        radius = 60
        ratio_cone_bottom_to_cylinder = 2

        Rz, Ry = GeometryPrimitives.calculate_zy_rotation_for_arrow(vec)
        mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=1/radius * ratio_cone_bottom_to_cylinder * scale,
            cone_height= size * ratio_cone_cylinder* scale,
            cylinder_radius=1/radius* scale,
            cylinder_height=size * (1 - ratio_cone_cylinder *scale))
        mesh.rotate(Ry, center=np.array([0, 0, 0]))
        mesh.rotate(Rz, center=np.array([0, 0, 0]))
        mesh.translate(origin)
        return(mesh)
    
    @staticmethod
    def createArrow(endpoints, radius=0.025/2, resolution=20, color=[0, 0, 0])->o3d.geometry.TriangleMesh:
        # mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=radius, cone_radius= radius * 2, cylinder_height=0.95, cone_height=0.05, resolution=resolution)
        # matrix = getCylinderTransform(endpoints)
        # mesh.transform(matrix)
        mesh = GeometryPrimitives.get_arrow(endpoints[1], endpoints[0], scale=1)
        mesh.compute_vertex_normals()
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([color]*len(mesh.vertices)))
        return mesh
    
    @staticmethod
    def createCoordinateBasis(basis, radius=0.025/2, resolution=20, color=[0, 0, 0])->o3d.geometry.TriangleMesh:
        meshes = []
        for i, b in enumerate(basis):
            mesh = GeometryPrimitives.createArrow(endpoints=[[0, 0, 0], b], radius=radius, resolution=resolution, color=color)
            meshes.append(mesh)
        mesh = GeometryPrimitives.collapseMeshObjects(meshes)
        return mesh
    
    def createMaxBasis(self, points, rgbs, lines): 
        for rgb, point in zip(rgbs, points): 
            self.add_obj(GeometryPrimitives.createSphere(center=point, color=rgb))
        for line in lines:
            self.add_obj(GeometryPrimitives.createCylinder(endpoints=[points[line[0]], points[line[1]]], color=[0, 0, 0]))

    def render(self):
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries(self.objects)
        return

class PSWrapper:

    # Enums
    class GeomFileType(Enum):
        OBJ = 1
        PLY = 2

    class ItemsToDisplay(Enum):
        MESH = 1
        LATTICE = 2
        BOTH = 3

    def __init__(self, observer, maxbasis, itemsToDisplay=ItemsToDisplay.MESH, displayBasis=DisplayBasisType.CONE, verbose=False):
        self.verbose = verbose
        self.observer = observer
        self.maxBasis = maxbasis
        self.dim = observer.dimension

        self.coneToBasis = self.maxBasis.get_cone_to_maxbasis_transform()
        self.HMatrix = getHeringMatrix(self.dim)

        self.dispBasisType = displayBasis
        self.itemsToDisplay = itemsToDisplay

        self.defaultInitPS()

        self.objects = []
        if itemsToDisplay == PSWrapper.ItemsToDisplay.MESH or itemsToDisplay == PSWrapper.ItemsToDisplay.BOTH:
            self.mesh = self.__createMesh(displayBasis)
        if itemsToDisplay == PSWrapper.ItemsToDisplay.LATTICE or itemsToDisplay == PSWrapper.ItemsToDisplay.BOTH:
            self.lattice = self.__createLattice(displayBasis)

    def add_obj(self, obj)->None:
        self.objects.append(obj)

    def defaultInitPS(self):
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_background_color([1, 1, 1, 0])
        ps.set_transparency_mode('pretty')
        self.ps = ps

    def loadPrinterGamut(self, filename, color=np.array([1, 0, 0])):
        gamut_data = np.load(filename)
        if self.dim > 3:
            basis_gamut = (self.HMatrix@self.coneToBasis@gamut_data.T).T[:, 1:]
        else: 
            basis_gamut = (self.HMatrix@self.coneToBasis@gamut_data.T).T

        alpha_mesh = alphashape.alphashape(basis_gamut[::1000, :], 3)
        
        colors = np.repeat([color], len(alpha_mesh.vertices), axis=0).reshape(-1, 3)
        ps_mesh = ps.register_surface_mesh("printer_mesh", np.asarray(alpha_mesh.vertices), np.asarray(alpha_mesh.faces), transparency=1, material='flat') 
        ps_mesh.add_color_quantity("printer_mesh_colors", colors, defined_on='vertices', enabled=True)
        ps_mesh.set_smooth_shade(True)

        return alpha_mesh.volume

    def __getIntermediateTransforms(self, num_steps):
        transforms = []
        rgb_in_lms = (np.linalg.inv(self.coneToBasis[::-1])@np.eye(self.dim))
        rgb = np.eye(self.dim)
        if self.dim > 3:
            rgb_in_lms =  (self.HMatrix@rgb_in_lms.T).T[1:, 1:]
            rgb = (self.HMatrix@rgb.T).T[1:, 1:]
            
        for t in np.arange(0, 1, 1/num_steps):
            # norm = np.linalg.norm(rgb_in_lms + easeFunction(1-t) * (rgb - rgb_in_lms), axis=1)
            new_3 = (rgb_in_lms + easeFunction(1-t) * (rgb - rgb_in_lms))
            mat = np.linalg.inv(new_3)
            mat4 = np.eye(4)
            mat4[:3, :3] = mat
            transforms += [mat4]
        return transforms
    
    def _getIntermediateTransformsLMSRGB(self, num_steps):
        transforms_LMS = []
        transforms_RGB = []
        # LMS to LMS in RGB space
        LMS = np.eye(self.dim)
        LMSinRGB = self.coneToBasis[::-1]@LMS

        RGBinLMS = (np.linalg.inv(self.coneToBasis[::-1])@np.eye(self.dim))
        RGB = np.eye(self.dim)

        for t in np.arange(0, 1, 1/num_steps):
            interpolateRGB = (RGBinLMS + easeFunction(t) * (RGB - RGBinLMS))
            # mat = np.linalg.inv(interpolateRGB)
            mat4 = np.eye(4)
            mat4[:3, :3] = interpolateRGB
            transforms_RGB += [mat4]
            interpolateLMS = (LMS + easeFunction(t) * (LMSinRGB - LMS))
            mat4 = np.eye(4)
            mat4[:3, :3] = interpolateLMS
            transforms_LMS += [mat4]
        return transforms_LMS, transforms_RGB
    
    def __getIntermediateTransformsOfSet(self, set1, set2, num_steps):
        transforms = []
        for t in np.arange(0, 1, 1/num_steps):
            new_3 = set1 + easeFunction(1-t) * (set2 - set1)
            mat = np.linalg.inv(new_3)
            mat4 = np.eye(4)
            mat4[:3, :3] = mat
            transforms += [mat4]
        return transforms

    def __createLattice(self, displayBasisType, displayBasis=None)->o3d.geometry.TriangleMesh:
        refs, points, rgbs, lines = self.maxBasis.getDiscreteRepresentation()
        if displayBasisType == DisplayBasisType.CONE:
            points = (np.linalg.inv(self.maxBasis.get_cone_to_maxbasis_transform())@points.T).T
        if self.dim > 3:
            points = (self.HMatrix@points.T).T[:, 1:]
        if displayBasis is not None:
            points = (displayBasis[:3, :3]@points.T).T
        
        p = GeometryPrimitives()
        p.createMaxBasis(points, rgbs, lines)
        [self.add_obj(obj) for obj in p.objects]
        lattice = o3d.geometry.TriangleMesh()
        for obj in p.objects:
            lattice += obj
        return lattice

    def __createMesh(self, displayBasisType)->o3d.geometry.TriangleMesh:

        # 1. Create point cloud using full_colors function
        if displayBasisType == DisplayBasisType.CONE:
            chrom_points, rgbs = self.observer.get_optimal_colors()
        elif displayBasisType == DisplayBasisType.MAXBASIS:
            T = self.maxBasis.get_cone_to_maxbasis_transform()
            chrom_points, rgbs = self.observer.get_optimal_colors()
            chrom_points = (T@(chrom_points.T)).T
            # chrom_points, rgbs = self.maxBasis.get_max_basis_observer().get_optimal_colors() # idk why this not working TODO: debug
        else:
            raise Exception(f"Invalid display basis type. Only support {list(DisplayBasisType)}")
        
        if self.dim > 3:
            chrom_points = (self.HMatrix@chrom_points.T).T[:, 1:]

        # 2. use open3d to process mesh since it computes vertex normals
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(chrom_points)
        mesh, point_indices = pcl.compute_convex_hull()
        mesh.compute_vertex_normals()
        
        # 3. set vertex colors
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs[point_indices])
        self.add_obj(mesh)
        
        return mesh
    

    def displayWTransparency(self, filepath=None):

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_background_color([1, 1, 1, 0])
        ps.set_transparency_mode('pretty')

        ps_mesh = ps.register_surface_mesh("mesh", np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles), transparency=0.8, material='flat') 
        ps_mesh.add_color_quantity("mesh_colors", np.asarray(self.mesh.vertex_colors), defined_on='vertices', enabled=True)
        ps_mesh.set_smooth_shade(True)
        if self.itemsToDisplay == PSWrapper.ItemsToDisplay.LATTICE or self.itemsToDisplay == PSWrapper.ItemsToDisplay.BOTH:
            ps_lattice = ps.register_surface_mesh("lattice", np.asarray(self.lattice.vertices), np.asarray(self.lattice.triangles), transparency=1) 
            ps_lattice.add_color_quantity("lattice_colors", np.asarray(self.lattice.vertex_colors), defined_on='vertices', enabled=True)
            ps_lattice.set_smooth_shade(True)

        if filepath is None:
            ps.show()
        else:
            ps.screenshot(filepath, True)
        return
    
    @staticmethod  
    def polarToCartesian(r, theta, phi):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)  
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    def renderObjectsPS(self, mesh_alpha=0.8, lattice_alpha=1, coord_alpha=1, matrix=None): 

        names = []

        if self.itemsToDisplay == PSWrapper.ItemsToDisplay.MESH or self.itemsToDisplay == PSWrapper.ItemsToDisplay.BOTH:
            
            ps_mesh = ps.register_surface_mesh("mesh", np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles), transparency=mesh_alpha, material='wax', smooth_shade=True) 
            ps_mesh.add_color_quantity("mesh_colors", np.asarray(self.mesh.vertex_colors), defined_on='vertices', enabled=True)
            names +=["mesh"]
            # ps_mesh.center_bounding_box()
        if self.itemsToDisplay == PSWrapper.ItemsToDisplay.LATTICE or self.itemsToDisplay == PSWrapper.ItemsToDisplay.BOTH:
            ps_lattice = ps.register_surface_mesh("lattice", np.asarray(self.lattice.vertices), np.asarray(self.lattice.triangles), transparency=lattice_alpha) 
            ps_lattice.add_color_quantity("lattice_colors", np.asarray(self.lattice.vertex_colors), defined_on='vertices', enabled=True)
            names +=["lattice"]
            # ps_lattice.set_transform(ps_mesh.get_transform())
        if matrix is not None:
            [ps.get_surface_mesh(s).set_transform(matrix) for s in names ]
        return names
    
    def _getmatrixBasisToLum(self): 
        hmatrix = self.HMatrix[::-1]
        HMat = np.eye(4)
        HMat[:3, :3] = hmatrix
        return HMat
    
    def renderPointCloud(self, points, rgbs, radius=0.001):
        points = self.ps.register_point_cloud("points", points)
        points.add_color_quantity("point_colors", rgbs, enabled=True)
        points.set_radius(radius, relative=False)
        return ["points"]
    
    def renderFlattenedMesh(self, name, item, matrix, arrow_plane_dist=1.5, mesh_alpha=0.8):
        new_verts = matrix[:3, :3]@(np.asarray(item.vertices).T)
        new_verts[2, :] = np.array([arrow_plane_dist]*len(new_verts.T))

        ps_mesh = ps.register_surface_mesh(name, new_verts.T, np.asarray(item.triangles), transparency=mesh_alpha, material='flat', smooth_shade=True) 
        ps_mesh.add_color_quantity(name + "_colors", np.asarray(item.vertex_colors), defined_on='vertices', enabled=True)
        ps_mesh.set_back_face_policy("cull")
        return ['flatmesh']
    
    def renderChromaticityTrichromat(self, mesh_alpha=0.8):
        new_verts = (self.HMatrix@np.asarray(self.mesh.vertices).T).T
        new_verts[:, 0] = np.array([0]*len(new_verts))
        new_rgb = np.array(new_verts)
        new_rgb[:, 0] = np.array([0.5]*len(new_rgb))
        new_rgb = (np.linalg.inv(self.HMatrix)@(new_rgb.T)).T

        ps_mesh = ps.register_surface_mesh("flatmesh", new_verts, np.asarray(self.mesh.triangles), transparency=mesh_alpha, material='flat', smooth_shade=True)
        ps_mesh.add_color_quantity("mesh_colors", np.clip(new_rgb, 0, 1), defined_on='vertices', enabled=True)
        ps_mesh.set_back_face_policy("cull")
        return ['flatmesh']

    def _getCoordBasis(self, name, vecs, coordAlpha=1):
        self.coordBasis = GeometryPrimitives.createCoordinateBasis(vecs, color=[0, 0, 0])
        ps_coord = ps.register_surface_mesh(f"{name}", np.asarray(self.coordBasis.vertices), np.asarray(self.coordBasis.triangles), transparency=coordAlpha) 
        ps_coord.add_color_quantity(f"{name}_colors", np.asarray(self.coordBasis.vertex_colors), defined_on='vertices', enabled=True)
        ps_coord.set_smooth_shade(True)
        return [f"{name}"]



    def assetRotationAroundLum(self, dirname, rotations_per_second, num_rotations, r, theta, mesh_alpha=0.8, lattice_alpha=1):
        names = self.renderObjectsPS(mesh_alpha=mesh_alpha, lattice_alpha=lattice_alpha)
        
        os.makedirs(dirname, exist_ok=True)
        frame_count = int(1/rotations_per_second * 30)
        total_frames = frame_count * num_rotations

        mat4 = np.eye(4)
        mat4[:3, :3] = np.flip(self.HMatrix, 0)
        coordBasis = mat4[:3, :3]@np.eye(3)
        [ps.get_surface_mesh(s).set_transform(mat4) for s in names ]
        
        self._getCoordBasis(coordBasis.T)

        for i in range(num_rotations):
            for j in range(frame_count): 
                iter = i * frame_count + j
                phi = 360 * j / frame_count
                point_3d = PSWrapper.polarToCartesian(r, theta, phi)

                ps.look_at(point_3d, [0, 0, np.sqrt(3)/2])
                ps.screenshot(dirname + f"/frame_{iter:03d}.png", True)

        exportAndPlay(dirname)


    def renderQArrow(self, radius=0.005, resolution=50, transparency=0.8):
        length = 1 * 0.05
        basisLMSQ = np.array([[0, 0, 1, 0]]) * length
        if self.dispBasisType == DisplayBasisType.MAXBASIS:
            basisLMSQ = ((self.HMatrix@self.coneToBasis)@basisLMSQ.T).T[:, 1:][0]
        else:
            basisLMSQ = (self.HMatrix@basisLMSQ.T).T[:, 1:][0]
        
        mesh = GeometryPrimitives.createCylinder(endpoints=[-basisLMSQ, basisLMSQ], radius=radius, resolution=resolution, color=[0, 0, 0])

        ps_line = ps.register_surface_mesh("qarrow", np.asarray(mesh.vertices), np.asarray(mesh.triangles), transparency=transparency, smooth_shade=True) 
        ps_line.add_color_quantity("qarrow_colors", np.asarray(mesh.vertex_colors), defined_on='vertices', enabled=True)
        return ["qarrow"]
    
    def __getGridOfArrows(self, top_arrow_dist=1,  arrow_plane_dist=1.25, num_points=8):
        width = 0.7
        y_values = np.linspace(-width, width, num_points) 
        z_values = np.linspace(-width, width, num_points) 
        y_mesh, z_mesh = np.meshgrid(y_values, z_values) 
        
        mesh = []
        for i in range(num_points):
            for j in range(num_points):
                mesh +=[GeometryPrimitives.createArrow(endpoints=[np.array([y_mesh[i, j], z_mesh[i, j], -top_arrow_dist]), np.array([y_mesh[i, j], z_mesh[i, j], arrow_plane_dist])], radius=0.025/10, resolution=20, color=[0, 0, 0])]
        mesh = GeometryPrimitives.collapseMeshObjects(mesh)
        ps_arrows = ps.register_surface_mesh("grid_arrows", np.asarray(mesh.vertices), np.asarray(mesh.triangles), transparency=1,smooth_shade=True) 
        ps_arrows.add_color_quantity("grid_arrows_colors", np.asarray(mesh.vertex_colors), defined_on='vertices', enabled=True)
        return ["grid_arrows"]
    
    def __getQArrowInQSpace(self, top_arrow_dist=1, arrow_plane_dist=1.25):
        mesh = GeometryPrimitives.createArrow(endpoints=[np.array([0, 0, -top_arrow_dist]), np.array([0, 0, arrow_plane_dist])], radius=0.025/10, resolution=20, color=[0, 0, 0])
        ps_arrows = ps.register_surface_mesh("grid_arrows", np.asarray(mesh.vertices), np.asarray(mesh.triangles), transparency=1,smooth_shade=True) 
        ps_arrows.add_color_quantity("grid_arrows_colors", np.asarray(mesh.vertex_colors), defined_on='vertices', enabled=True)
        return ["grid_arrows"]
    
    def _getTransformQUpDir(self): # TODO: Erase code to work with new MaxBasis getMatrixOrientationQUp method
        length = 1
        basisLMSQ = np.array([[0, 0, 1, 0]]) * length # Q cone
        if self.dispBasisType == DisplayBasisType.MAXBASIS:
            basisLMSQ = ((self.HMatrix@self.coneToBasis)@basisLMSQ.T).T[:, 1:][0]
        else:
            basisLMSQ = (self.HMatrix@basisLMSQ.T).T[:, 1:][0]
        
        matrix = np.linalg.inv(getCylinderTransform([np.array([0, 0, 0]), basisLMSQ/np.linalg.norm(basisLMSQ)])) # go from Q to the basis [1, 0, 0]
        return matrix
    
    def __getLineOfArrows(self, pos_line, num_points=8): 
        
        width = 0.8
        y_values = np.linspace(-width, width, num_points)
        
        mesh = []
        for i in range(num_points):
            mesh +=[GeometryPrimitives.createArrow(endpoints=[np.array([0, y_values[i], -length]), np.array([0, y_values[i], pos_line])], radius=0.025/10, resolution=20, color=[0, 0, 0])]
        
        mesh = GeometryPrimitives.collapseMeshObjects(mesh)
        ps_arrows = ps.register_surface_mesh("grid_arrows", np.asarray(mesh.vertices), np.asarray(mesh.triangles), transparency=1,smooth_shade=True) 
        ps_arrows.add_color_quantity("grid_arrows_colors", np.asarray(mesh.vertex_colors), defined_on='vertices', enabled=True)

    def figureGridOfArrows(self, top_arrow_dist, arrow_plane_dist, z, theta, r, mesh_alpha=0.8, lattice_alpha=1):

        self.renderObjectsPS(mesh_alpha=mesh_alpha, lattice_alpha=lattice_alpha)
        self.__getGridOfArrows(top_arrow_dist, arrow_plane_dist, num_points=6)
        matrix = self._getTransformQUpDir()
        self.renderFlattenedMesh("flatmesh", self.mesh, matrix, arrow_plane_dist=arrow_plane_dist, mesh_alpha=1)
        ps.set_up_dir("neg_z_up")
        ps.set_front_dir("x_front")
        ps.set_ground_plane_mode("none")
        ps.set_SSAA_factor(4)
        ps.get_surface_mesh("mesh").set_transform(matrix)

        point_3d = PSWrapper.polarToCartesian(r, 90, theta)
        ps.look_at([point_3d[0], point_3d[1], -z], [0, 0, -z])
        # ps.screenshot(filename, transparent_bg=True)
        # subprocess.call(["open", filename])
        # ps.show()

    def figureBallProjection(self, arrow_plane_dist=1.35, mesh_alpha=0.8): 
        # self.renderObjectsPS(mesh_alpha=mesh_alpha)
        # self.__getGridOfArrows(1.1, arrow_plane_dist, num_points=6)
        # matrix = self._getTransformQUpDir()
        # self.renderFlattenedMesh("flatmesh", self.mesh, matrix, arrow_plane_dist=arrow_plane_dist, mesh_alpha=1)
        # ps.set_up_dir("neg_z_up")
        # ps.set_front_dir("x_front")
        # ps.set_ground_plane_mode("none")
        # ps.set_SSAA_factor(4)
        # ps.get_surface_mesh("mesh").set_transform(matrix)
        
        self.renderObjectsPS(mesh_alpha=mesh_alpha)
        matrix = self._getTransformQUpDir()
        self.__getQArrowInQSpace(top_arrow_dist=1.1, arrow_plane_dist=arrow_plane_dist)
        self.renderFlattenedMesh("flatmesh", self.mesh, matrix, arrow_plane_dist=arrow_plane_dist, mesh_alpha=1)
        ps.set_up_dir("neg_z_up")
        ps.set_front_dir("x_front")
        ps.set_ground_plane_mode("none")
        ps.set_SSAA_factor(4)
        ps.get_surface_mesh("mesh").set_transform(matrix)

        # render ball
        ball = GeometryPrimitives.createSphere(center=[0, 0, 0], radius=0.1, color=[0, 0, 0])
        ps_ball = ps.register_surface_mesh("ball", np.asarray(ball.vertices), np.asarray(ball.triangles), transparency=1,smooth_shade=True)
        ps_ball.add_color_quantity("ball_colors", np.asarray(ball.vertex_colors), defined_on='vertices', enabled=True)
        ps_ball.set_transform(matrix)

        self.renderFlattenedMesh("flatball", ball, matrix, arrow_plane_dist=arrow_plane_dist-0.02, mesh_alpha=1)


        point_3d = PSWrapper.polarToCartesian(5, 90, 180)
        ps.look_at([point_3d[0], point_3d[1], -0], [0, 0, -0])
    
        return

    def assetRotateAroundZ(self, dirname, rotations_per_second, num_rotations, r, theta, mesh_alpha=0.8, lattice_alpha=1): 
        names = self.renderObjectsPS(mesh_alpha=mesh_alpha, lattice_alpha=lattice_alpha)
        # names += self.renderQArrow()
        # matrix = self.__getGridOfArrows()
        matrix = self._getTransformQUpDir()
        os.makedirs(dirname, exist_ok=True)
        frame_count = int(1/rotations_per_second * 30)
        total_frames = frame_count * num_rotations
        height = r * np.cos(theta)
        for i in range(num_rotations):
            for j in range(frame_count): 
                iter = i * frame_count + j
                [self.ps.get_surface_mesh(s).set_transform(matrix) for s in names ]
                # ps.get_surface_mesh("mesh").set_transform(matrix)
                # ps.get_surface_mesh("qarrow").set_transform(matrix)
                
                phi = 360 * j / frame_count
                point_3d = PSWrapper.polarToCartesian(r, theta, phi)
                # move point down to baseline over the course of the animation
                # point_3d -= np.array([0, 0, height * easeFunction(iter/total_frames)])
                ps.look_at(point_3d, [0, 0, 0])
                ps.screenshot(dirname + f"/frame_{iter:03d}.png", True)
        exportAndPlay(dirname)
        return
    
    def sliceThrough(self, dirname):
        self.renderObjectsPS(mesh_alpha=1)
        # Add a slice plane
        ps_plane = self.ps.add_scene_slice_plane()
        ps_plane.set_draw_plane(False) # render the semi-transparent gridded plane
        ps_plane.set_draw_widget(False)

        os.makedirs(dirname, exist_ok=True)
        # Animate the plane sliding along the scene
        for iter, t in enumerate(np.linspace(0., 2*np.pi, 300)):
            pos = np.cos(t) * np.sqrt(3)
            ps_plane.set_pose((0., 0., pos), (-1., -1., -1.))

            # Take a screenshot at each frame
            self.ps.screenshot(dirname + f"/frame_{iter:03d}.png", True)
        exportAndPlay(dirname)
        return
    
    def assetRotatePointCloud(self, dirname, rotations_per_second, num_rotations, r, theta, mesh_alpha=0.8, lattice_alpha=1): 
        self.renderObjectsPS(mesh_alpha=mesh_alpha, lattice_alpha=lattice_alpha)
        self.renderQArrow()
        matrix = self.__getGridOfArrows()
        os.makedirs(dirname, exist_ok=True)
        frame_count = int(1/rotations_per_second * 30)
        total_frames = frame_count * num_rotations
        height = r * np.cos(theta)
        for i in range(num_rotations):
            for j in range(frame_count): 
                iter = i * frame_count + j
                ps.get_surface_mesh("mesh").set_transform(matrix)
                ps.get_surface_mesh("qarrow").set_transform(matrix)
                
                phi = 360 * j / frame_count
                point_3d = PSWrapper.polarToCartesian(r, theta, phi)
                # move point down to baseline over the course of the animation
                # point_3d -= np.array([0, 0, height * easeFunction(iter/total_frames)])
                ps.look_at(point_3d, [0, 0, 0])
                ps.screenshot(dirname + f"/frame_{iter:03d}.png", True)
        exportAndPlay(dirname)
        return

    
    def assetCoordinateBasisTransform(self, dirname, rotations_per_second, r, theta, mesh_alpha=0.8, lattice_alpha=1, phi_offset=0): 
        HMat = self._getmatrixBasisToLum()
        self.renderObjectsPS(mesh_alpha=mesh_alpha, lattice_alpha=lattice_alpha, matrix=HMat)

        os.makedirs(dirname, exist_ok=True)
        frame_count = int(1/rotations_per_second * 30)
        transforms = self.__getIntermediateTransforms(frame_count)

        ps_mesh = ps.get_surface_mesh("mesh")
        ps_lattice = ps.get_surface_mesh("lattice")

    
        def rotate_once(offset, frame_count):
            for j in range(frame_count): # rotate once
                phi = 360 * (j + phi_offset) / frame_count
                point_3d = PSWrapper.polarToCartesian(r, theta, phi)
                ps.look_at(point_3d, [0, 0, np.sqrt(3)/2])
                ps.screenshot(dirname + f"/frame_{offset * frame_count + j:03d}.png", True)

        def expandBasis(offset, frame_count):
            for j in range(frame_count): # rotate back
                phi = 360 * (phi_offset) / frame_count
                point_3d = PSWrapper.polarToCartesian(r, theta, phi)
                # add coordinate transform

                ps_mesh.set_transform(HMat@transforms[j])
                lattice = self.__createLattice(DisplayBasisType.CONE, HMat@transforms[j])
                ps_lattice = ps.register_surface_mesh("moving_lattice", np.asarray(lattice.vertices), np.asarray(lattice.triangles), transparency=lattice_alpha) 
                ps_lattice.add_color_quantity("moving_lattice_colors", np.asarray(lattice.vertex_colors), defined_on='vertices', enabled=True)
                ps_lattice.set_smooth_shade(True)

                ps.look_at(point_3d, [0, 0, np.sqrt(3)/2])
                ps.screenshot(dirname + f"/frame_{offset * frame_count + j:03d}.png", True)
        
        
        rotate_once(0, frame_count)
        ps_lattice.remove()
        expandBasis(1, frame_count)
        rotate_once(2, frame_count)

        exportAndPlay(dirname)
        return
    
    def assetCoordinateTransformLMSRGB(self, dirname, rotations_per_second, r, theta, mesh_alpha=0.8, lattice_alpha=1, phi_offset=0): 
        HMat = self._getmatrixBasisToLum()
        self.renderObjectsPS(mesh_alpha=mesh_alpha, lattice_alpha=lattice_alpha, matrix=HMat)

        os.makedirs(dirname, exist_ok=True)
        frame_count = int(1/rotations_per_second * 30)
        transforms_LMS, transforms_RGB = self._getIntermediateTransformsLMSRGB(frame_count)

        ps_mesh = ps.get_surface_mesh("mesh")
        names = self._getCoordBasis("RGB_coords", ((HMat@transforms_RGB[0])[:3, :3]).T, 1)
        names += self._getCoordBasis("LMS_coords", ((HMat@transforms_LMS[0])[:3, :3]).T, 1)

        def rotate_once(offset, frame_count):
            for j in range(frame_count): # rotate once
                phi = 360 * (j + phi_offset) / frame_count
                point_3d = PSWrapper.polarToCartesian(r, theta, phi)
                ps.look_at(point_3d, [0, 0, np.sqrt(3)/2])
                ps.screenshot(dirname + f"/frame_{offset * frame_count + j:03d}.png", True)

        def expandBasis(offset, frame_count):
            for j in range(frame_count): # rotate back
                [ps.get_surface_mesh(n).remove() for n in names]

                phi = 360 * (phi_offset) / frame_count
                point_3d = PSWrapper.polarToCartesian(r, theta, phi)
                # add coordinate transform

                ps_mesh.set_transform(HMat@transforms_LMS[j])
                self._getCoordBasis("RGB_coords", ((HMat@transforms_RGB[j])[:3, :3]).T)
                self._getCoordBasis("LMS_coords", ((HMat@transforms_LMS[j])[:3, :3]).T)

                ps.look_at(point_3d, [0, 0, np.sqrt(3)/2])
                ps.screenshot(dirname + f"/frame_{offset * frame_count + j:03d}.png", True)
        
        
        rotate_once(0, frame_count)
        expandBasis(1, frame_count)
        rotate_once(2, frame_count)

        exportAndPlay(dirname)
        return
    

    """
    exportMesh - Export the mesh to a file in the specified format, 
    OBJ is meant for keynote export, as usdzconvert only takes in OBJ files.
    Default export mode is ply.
    The file will contain the vertex positions, normals, faces, and colors of the mesh.
    """
    def exportMesh(self, filepath, type=GeomFileType.PLY)->None:
        # concatenate all the mesh parts together
        mesh = o3d.geometry.TriangleMesh()
        for obj in self.objects:
            mesh += obj

        if type == PSWrapper.GeomFileType.OBJ:
            filepath += ".obj"
            with open(filepath, 'w') as f:
                f.write("# OBJ file\n")
                # first write vertex position and vertex colors
                for (v, c) in zip(mesh.vertices, mesh.vertex_colors):
                    f.write("v {0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}\n".format(v[0], v[1], v[2], c[0], c[1], c[2]))
                # then write vertex normals
                for n in mesh.vertex_normals:
                    f.write("vn {0:.4f} {1:.4f} {2:.4f}\n".format(n[0], n[1], n[2]))
                for p in mesh.triangles:
                    f.write("f {0:d} {1:d} {2:d}\n".format(int(p[0]+1), int(p[1]+1), int(p[2]+1)))
            print("Exported mesh to", filepath)
            # your terminal must have usdzconvert on the PATH
            subprocess.run(["usdzconvert", filepath, filepath.replace(".obj", ".usdz")]) 
            return
        
        elif type == PSWrapper.GeomFileType.PLY:
            filepath += ".ply"
            o3d.io.write_triangle_mesh(filepath, mesh)
            
        else:
            raise Exception(f"Invalid export type. Only support {list(PSWrapper.GeomFileType)}")
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
