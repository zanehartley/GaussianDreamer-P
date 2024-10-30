from dataclasses import dataclass, field
import torch
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from argparse import ArgumentParser, Namespace
import os
import copy
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
import numpy as np
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
import io  
from PIL import Image  
import open3d as o3d

from torchvision.transforms.functional import equalize

from sklearn.preprocessing import MinMaxScaler
import cv2
   
def histogram_equalization_pytorch(depth_map):
    # Normalize the depth map to the range [0, 1]
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    # Calculate the cumulative distribution function (CDF)
    cdf = torch.cumsum(normalized_depth.histc(), dim=0)
    # Normalize the CDF to the range [0, 1]
    normalized_cdf = cdf / cdf[-1]
    # Map the normalized depth values to the normalized CDF
    equalized_depth = normalized_cdf[normalized_depth.long()]
    return equalized_depth

def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


@threestudio.register("gaussiandreamer-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        sh_degree: int = 0
        load_type: int = 0
        load_path: str = "./load/shapes/stand.obj"



    cfg: Config
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.sh_degree =self.cfg.sh_degree
        self.load_type =self.cfg.load_type
        self.load_path = self.cfg.load_path

        self.gaussian = GaussianModel(sh_degree = self.sh_degree)
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float16, device="cuda")
        
        # The minimum depth intensity for pixels that are part of the object
        #(ensures that parts of the object do not blend into the background)
        self.min_depth_intensity = 0.1
 
        # Scale the object depth for better appearance and distinction from the background
        self.depth_intensity_scale = 1.75
 
        # The value that determines if the depth is part of the background
        self.depth_background_cutoff = 0.97

    
    def save_gif_to_file(self,images, output_file):  
        with io.BytesIO() as writer:  
            images[0].save(  
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
            )  
            writer.seek(0)  
            with open(output_file, 'wb') as file:  
                file.write(writer.read())
    
    def scale_to_longest_dimension(self, coords, target_range=(-0.9, 0.9)):


        mean_coords = np.mean(coords, axis=0)
        print(f"Center is: {mean_coords}")
        coords -= mean_coords
        print(f"Center is: {mean_coords}")

        # Calculate the range of each dimension
        ranges = np.max(coords, axis=0) - np.min(coords, axis=0)

        # Find the index of the longest dimension
        longest_dim_idx = np.argmax(ranges)

        # Calculate the scaling factor for the longest dimension
        scaling_factor = (target_range[1] - target_range[0]) / ranges[longest_dim_idx]

        # Scale all dimensions proportionally
        scaled_coords = coords * scaling_factor

        return scaled_coords
            
    def calculate_erosion_kernal(self):
            """
            Calculate the size of the erosion kernel based on the camera's distance from the object.
    
            The erosion kernel determines the 'harshness' of the edge removal during depth map processing.
            Closer camera is to the object, the 'harsher' the kernel needs to be
    
            Returns:
                int: The size of the erosion kernel. Possible values are 3, 5, or 7 depending on the camera's radius.
            """
            # Calculate erosion kernal (this determines the 'harshness' of the edge removal)
            # Closer camera is to the object, the 'harsher' the kernel needs to be
            erode_kernal_size = 7
            if self.cfg.radius < 2.0:
                erode_kernal_size = 3
            elif self.cfg.radius < 4.0:
                erode_kernal_size = 5
    
            return erode_kernal_size
   
    def normalise_depth_maps(self, depth_maps):
        """
        This function normalizes depth maps by identifying and separating the background from the foreground,
        rescaling the depth intensities, and applying dilation and erosion to reduce noise and remove erroneous edges.

        Args:
            depth_maps (torch.Tensor): The input depth maps to be normalized and processed.

        Returns:
            torch.Tensor: The processed and normalized depth maps with refined depth information.
        """

        depths_np = depth_maps.detach().cpu().numpy()

        # Determine background from foreground
        depths_np = np.round(depths_np, 2)
        unique, counts = np.unique(depths_np, return_counts=True)
        start_bin = unique[1]
        end_bin = unique[-1]

        # Invert image and ensure that background is still set to 0
        normalised_depths =  1.0 - ((np.maximum((depths_np-start_bin), np.full(depths_np.shape, 0)))/(end_bin-start_bin))
        normalised_depths = np.where((normalised_depths < self.depth_background_cutoff), normalised_depths, 0)

        # Rescale the image to ensure that depth of object has a minumum intensity
        normalised_depths = np.minimum(((normalised_depths + self.min_depth_intensity) * self.depth_intensity_scale), np.full(depths_np.shape, 1))
        normalised_depths = np.where((normalised_depths > (self.min_depth_intensity * self.depth_intensity_scale)), normalised_depths, 0)

        eroded_depths = np.array([])
        for normalised_depth in normalised_depths:

            # Dilate depth to ensure that object depth is uniform (removes noise)
            #kernel = np.ones((3,3), np.uint8)
            #normalised_depth = cv2.dilate(normalised_depth, kernel, iterations=1)

            # Remove erroneous edges from depth of object
            erode_kernal_size = self.calculate_erosion_kernal()

            kernel = np.ones((erode_kernal_size, erode_kernal_size), np.uint8)
            normalised_depth = cv2.erode(normalised_depth, kernel, iterations=1)

            """kernel = np.ones((erode_kernal_size-2, erode_kernal_size-2), np.uint8)
            normalised_depth = cv2.erode(normalised_depth, kernel, iterations=1)"""

            normalised_depth = np.expand_dims(normalised_depth, axis=0)

            if len(eroded_depths) == 0:
                eroded_depths = normalised_depth
            else:
                eroded_depths = np.concatenate((eroded_depths, normalised_depth))

        eroded_depths = np.expand_dims(eroded_depths, axis=3)

        return torch.from_numpy(eroded_depths)
   
    def normalise_depth_maps(self, depth_maps):
        """
        This function normalizes depth maps by identifying and separating the background from the foreground,
        rescaling the depth intensities, and applying dilation and erosion to reduce noise and remove erroneous edges.
 
        Args:
            depth_maps (torch.Tensor): The input depth maps to be normalized and processed.
 
        Returns:
            torch.Tensor: The processed and normalized depth maps with refined depth information.
        """
 
        depths_np = depth_maps.detach().cpu().numpy()
 
        # Determine background from foreground
        depths_np = np.round(depths_np, 2)
        unique, counts = np.unique(depths_np, return_counts=True)
        start_bin = unique[1]
        end_bin = unique[-1]
 
        # Invert image and ensure that background is still set to 0
        normalised_depths =  1.0 - ((np.maximum((depths_np-start_bin), np.full(depths_np.shape, 0)))/(end_bin-start_bin))
        normalised_depths = np.where((normalised_depths < self.depth_background_cutoff), normalised_depths, 0)
 
        # Rescale the image to ensure that depth of object has a minumum intensity
        normalised_depths = np.minimum(((normalised_depths + self.min_depth_intensity) * self.depth_intensity_scale), np.full(depths_np.shape, 1))
        normalised_depths = np.where((normalised_depths > (self.min_depth_intensity * self.depth_intensity_scale)), normalised_depths, 0)
 
        eroded_depths = np.array([])
        for normalised_depth in normalised_depths:
 
            # Dilate depth to ensure that object depth is uniform (removes noise)
            #kernel = np.ones((3,3), np.uint8)
            #normalised_depth = cv2.dilate(normalised_depth, kernel, iterations=1)
 
            # Remove erroneous edges from depth of object
            erode_kernal_size = self.calculate_erosion_kernal()
 
            kernel = np.ones((erode_kernal_size, erode_kernal_size), np.uint8)
            normalised_depth = cv2.erode(normalised_depth, kernel, iterations=1)
 
            kernel = np.ones((erode_kernal_size-2, erode_kernal_size-2), np.uint8)
            normalised_depth = cv2.erode(normalised_depth, kernel, iterations=1)
 
            normalised_depth = np.expand_dims(normalised_depth, axis=0)
 
            if len(eroded_depths) == 0:
                eroded_depths = normalised_depth
            else:
                eroded_depths = np.concatenate((eroded_depths, normalised_depth))
 
        eroded_depths = np.expand_dims(eroded_depths, axis=3)
 
        return torch.from_numpy(eroded_depths)

    def load_ply_and_get_data(self, filename):
        # Load the PLY file using Open3D
        point_cloud = o3d.io.read_point_cloud(filename)
        
        voxel_size = 0.001  # Adjust voxel size as needed
        point_cloud = point_cloud.voxel_down_sample(voxel_size)

        # Get coordinates from the point cloud
        coords = np.asarray(point_cloud.points)

        # Print max and min values before scaling
        print("Before scaling:")
        print("X-axis: min =", coords[:, 0].min(), ", max =", coords[:, 0].max())
        print("Y-axis: min =", coords[:, 1].min(), ", max =", coords[:, 1].max())
        print("Z-axis: min =", coords[:, 2].min(), ", max =", coords[:, 2].max())

        # Scale coordinates
        #scaler = MinMaxScaler(feature_range=(-0.9, 0.9))  # Set the scaling range
        #coords = scaler.fit_transform(coords)
        coords = self.scale_to_longest_dimension(coords)

        # Print max and min values after scaling
        print("After scaling:")
        print("X-axis: min =", coords[:, 0].min(), ", max =", coords[:, 0].max())
        print("Y-axis: min =", coords[:, 1].min(), ", max =", coords[:, 1].max())
        print("Z-axis: min =", coords[:, 2].min(), ", max =", coords[:, 2].max())

        # Check if RGB color information exists
        '''
        if point_cloud.has_colors():
            rgb = np.asarray(point_cloud.colors)
        else:
            # Handle the case where no colors are present (return dummy values or None)
            #rgb = np.zeros((coords.shape[0], 3))  # Example: fill with black (0, 0, 0)
            green_values = np.random.rand(coords.shape[0]) * 0.5 + 0.25  # Random between 0.25 and 0.75
            zeros = np.zeros(coords.shape[0])
            rgb = np.stack((zeros, green_values, zeros), axis=-1)
        '''
        rgb = np.full(coords.shape, 0.5)
        return coords, rgb, 0.4  # You can return additional data in the third slot


    def shape(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
        diffusion = diffusion_from_config_shape(load_config('diffusion'))

        batch_size = 1
        guidance_scale = 15.0
        prompt = str(self.cfg.prompt_processor.prompt)
        print('prompt',prompt)

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        render_mode = 'nerf' # you can change this to 'stf'
        size = 256 # this is the size of the renders; higher values take longer to render.

        cameras = create_pan_cameras(size, device)

        self.shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)

        pc = decode_latent_mesh(xm, latents[0]).tri_mesh()


        skip = 1
        coords = pc.verts
        rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

        coords = coords[::skip]
        rgb = rgb[::skip]

        self.num_pts = coords.shape[0]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)
        self.point_cloud = point_cloud

        return coords,rgb,0.4
    
    def add_points(self,coords,rgb):
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        

        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        num_points = 1000000  
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))


        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)


        points_inside = []
        color_inside= []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01 * 3:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        #print("==================================================")
        #print(f"all_coords: {all_coords.shape}")
        #print(f"coords: {coords.shape}")
        #print("==================================================")
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        return all_coords,all_rgb

    def smpl(self):
        self.num_pts  = 50000
        mesh = o3d.io.read_triangle_mesh(self.load_path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=self.num_pts)
        coords = np.array(point_cloud.points)
        shs = np.random.random((self.num_pts, 3)) / 255.0
        rgb = SH2RGB(shs)
        adjusment = np.zeros_like(coords)
        adjusment[:,0] = coords[:,2]
        adjusment[:,1] = coords[:,0]
        adjusment[:,2] = coords[:,1]
        current_center = np.mean(adjusment, axis=0)
        center_offset = -current_center
        adjusment += center_offset
        return adjusment,rgb,0.5
    
    def pcb(self):
        # Since this data set has no colmap data, we start with random points
        if self.load_type==0:
            coords,rgb,scale = self.shape()
        elif self.load_type==1:
            coords,rgb,scale = self.smpl()
        elif self.load_type==2:
            filename = "./inputs/duel_bean_1.ply"
            coords, rgb, scale = self.load_ply_and_get_data(filename)
        else:
            raise NotImplementedError
        
        bound= self.radius*scale
        #all_coords,all_rgb = self.add_points(coords,rgb)
        all_coords,all_rgb = coords,rgb

        pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((all_coords.shape[0], 3)))

        return pcd
    
    
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        self.viewspace_point_list = []
        for id in range(batch['c2w_3dgs'].shape[0]):
       
            viewpoint_cam  = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])
            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground.to(torch.float32))
            render_pkg_for_depth = render(viewpoint_cam, self.gaussian_copy, self.pipe, renderbackground.to(torch.float32))
            image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii,self.radii)
                
            #depth = render_pkg["depth_3dgs"]
            depth = render_pkg_for_depth["depth_3dgs"]
            depth =  depth.permute(1, 2, 0)
            
            image =  image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        
        self.visibility_filter = self.radii>0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
    
    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)
        
        if self.true_global_step > 600:
            self.guidance.set_min_max_steps(min_step_percent=0.12, max_step_percent=0.35)
            #self.gaussian._xyz.requres_grad = False
        if self.true_global_step > 1000:
            self.guidance.set_min_max_steps(min_step_percent=0.12, max_step_percent=0.25)
        if self.true_global_step > 2000:
            self.guidance.set_min_max_steps(min_step_percent=0.075, max_step_percent=0.15)

        self.gaussian.update_learning_rate(self.true_global_step)

        #This step seems to render the image from the gaussian splat
        out = self(batch)

        prompt_utils = self.prompt_processor()
        #This step then gets the image from the gaussian splat render
        images = out["comp_rgb"]
        depths = out["depth"].detach()
        normalised_depths = self.normalise_depth_maps(depths)

        depth_np = normalised_depths[0].detach().cpu().numpy()
        cv2.imwrite("./test_depth.png", depth_np * 255/np.max(depth_np))

        for i in range(images.shape[0]):
            cv2.imwrite(f"./outputs/test_depth{i}_{self.true_global_step}.png", normalised_depths[i].numpy()*255)
            cv2.imwrite(f"./outputs/test_rgb{i}_{self.true_global_step}.png", images[i].detach().cpu().numpy()*255)

        guidance_eval = (self.true_global_step % 200 == 0)
        # guidance_eval = False
        
        ########################## 2D Diffusion Step #############################
        #This step seems to actually do the 2D diffusion and perhaps also the comparison to the real image.
        guidance_out = self.guidance(
            images, normalised_depths, prompt_utils, **batch, rgb_as_latents=False,guidance_eval=guidance_eval
        )

        loss = 0.0

        loss = loss + guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])
        
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):

        with torch.no_grad():
            
            if self.true_global_step < 900: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > 300 and self.true_global_step % 100 == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 500 else None # 3000
                    self.gaussian.densify_and_prune(0.0002 , 0.05, self.cameras_extent, size_threshold) 

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if False else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float16, device="cuda")

        out = self(batch,testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last_3dgs.ply")
        self.gaussian.save_ply(save_path)
        # self.pointefig.savefig(self.get_save_path("pointe.png"))
        if self.load_type==0:
            o3d.io.write_point_cloud(self.get_save_path("shape.ply"), self.point_cloud)
            self.save_gif_to_file(self.shapeimages, self.get_save_path("shape.gif"))
        load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-test-color.ply"))
        
    def cull_large_gaussians(self, cull_std_factor=0):
 
        # Calculate Gaussian Volumes
        gaussian_sizes = torch.sum(self.gaussian.get_scaling, axis=1)
 
        # Reoder Gaussians based on volume
        sorted_sizes, sorted_indices = torch.sort(gaussian_sizes)

        #print(self.gaussian.get_scaling.shape)
 
        # Get the mean and std of the gaussian sizes
        mean_and_std = torch.std_mean(sorted_sizes)
 
        # Calculate outliers based on the std multiplied by the cull factor
        outlier_range = (mean_and_std[0] * cull_std_factor) + mean_and_std[1]
        max_gaussian_size = sorted_indices[sorted_sizes > outlier_range][0]
        culled_gaussians = sorted_indices < max_gaussian_size

        # Prune large Gaussians
        self.gaussian.prune_points(culled_gaussians)

        print(self.gaussian.get_scaling.shape)

        exit(0)

    def get_max_radius(self, radius_factor=1.3):
       
        with torch.no_grad():
            positions = self.gaussian.get_xyz.clone()
 
            euclid_distances = positions.pow(2).sum(1).sqrt()
 
            return torch.absolute(torch.max(euclid_distances)) * radius_factor
 
    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
       
        opt = OptimizationParams(self.parser)
        point_cloud = self.pcb()
        self.cameras_extent = 4.0
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)
 
        self.pipe = PipelineParams(self.parser)
        self.gaussian.training_setup(opt)
 
        # Cull large Gaussians
        self.cull_large_gaussians()
 
        optimal_camera_distance = self.get_max_radius()
       
        print()
        print()
        print("!!!!!!!!!!!!!!!! OPTIMAL CAMERA DISTANCES !!!!!!!!!!!!!!!!")
        print()
        print("eval_camera_distance:")
        print(optimal_camera_distance.item())
        print()
        print("camera_distance_range:")
        print([(optimal_camera_distance - (optimal_camera_distance/10)).item(),
               (optimal_camera_distance + (optimal_camera_distance/10)).item()])
        print()
        print()
 
        self.gaussian_copy = copy.deepcopy(self.gaussian)
 
        ret = {
            "optimizer": self.gaussian.optimizer,
        }
 
        return ret
 
        # Cull large Gaussians
    def cull_large_gaussians(self, cull_std_factor=3):
 
        # Calculate Gaussian Volumes
        gaussian_sizes = torch.sum(self.gaussian.get_scaling, axis=1)
 
        # Reoder Gaussians based on volume
        sorted_sizes, sorted_indices = torch.sort(gaussian_sizes)
 
        # Get the mean and std of the gaussian sizes
        mean_and_std = torch.std_mean(sorted_sizes)
 
        # Calculate outliers based on the std multiplied by the cull factor
        outlier_range = (mean_and_std[0] * cull_std_factor) + mean_and_std[1]
        max_gaussian_size = sorted_indices[sorted_sizes > outlier_range][0]
        culled_gaussians = sorted_indices < max_gaussian_size
       
        # Prune large Gaussians
        self.gaussian.prune_points(culled_gaussians)