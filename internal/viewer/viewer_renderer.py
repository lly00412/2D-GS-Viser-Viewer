import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
# from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings, ExtendedSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    
    # gradient magnitude
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2).norm(dim=0, keepdim=True)
    return magnitude

class ViewerRenderer:
    def __init__(self,
                gaussian_model,
                pipe,
                background_color,
                splat_args: ExtendedSettings = None,
                do_initialize=True):
        super().__init__()
        self.gaussian_model = gaussian_model
        self.background_color = background_color
        self.clm_colors = torch.tensor(plt.cm.get_cmap("turbo").colors, device="cuda")
        self.pipe = pipe
        self.splat_args = splat_args
        if do_initialize:
            self.update_pc_features()

    def update_pc_features(self):
        self.means3D = self.gaussian_model.get_xyz
        self.all_ids = torch.ones(self.means3D.shape[0], dtype=torch.bool, device=self.means3D.device)
        screenspace_points = torch.zeros_like(self.means3D, dtype=self.means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        self.means2D = screenspace_points
        self.opacity = self.gaussian_model.get_opacity
        self.scales = self.gaussian_model.get_scaling
        self.rotations = self.gaussian_model.get_rotation
        self.shs = self.gaussian_model.get_features

    def disk_kernel(self, opacity):
        return torch.exp(-1/2 * 100 * torch.clamp(opacity-0.5, min=0) ** 2)

    def color_map(self, map):
        if not map.min() == map.max():
            map = (map - map.min()) / (map.max() - map.min())
            map = (map * 255).round().long().squeeze()
            map = self.clm_colors[map].permute(2, 0, 1)
            return map
        else:
            map = torch.zeros_like(map, device=map.device).round().long().squeeze()
            map = self.clm_colors[map].permute(2, 0, 1)
            return map

    def render_viewer(self,
                      viewpoint_camera,
                      active_sh_degree,
                      bg_color: torch.Tensor,
                      scaling_modifier=1.0,
                      override_color=None,
                      sparsity: int = 1,
                      show_ptc: bool = False,
                      show_disk: bool = False,
                      point_size: float = 0.001,
                      valid_range=None
                      ):
        """
        Render the scene. 
        Background tensor (bg_color) must be on GPU!
        """

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings_rgb = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            inv_viewprojmatrix=viewpoint_camera.full_proj_transform_inverse,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            settings=self.splat_args,
            render_depth=False,
            debug=self.pipe.debug
        )
        rasterizer_rgb = GaussianRasterizer(raster_settings=raster_settings_rgb)

        raster_settings_d = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            inv_viewprojmatrix=viewpoint_camera.full_proj_transform_inverse,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            settings=self.splat_args,
            render_depth=True,
            debug=self.pipe.debug
        )
        rasterizer_depth = GaussianRasterizer(raster_settings=raster_settings_d)

        if valid_range is not None:
            is_x_in_range = (valid_range[0][0] <= self.means3D[:, 0]) & (self.means3D[:, 0] <= valid_range[0][1])
            is_y_in_range = (valid_range[1][0] <= self.means3D[:, 1]) & (self.means3D[:, 1] <= valid_range[1][1])
            is_z_in_range = (valid_range[2][0] <= self.means3D[:, 2]) & (self.means3D[:, 2] <= valid_range[2][1])
            is_in_box = is_x_in_range & is_y_in_range & is_z_in_range
        else:
            is_in_box = self.all_ids

            # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
            # scaling / rotation by the rasterizer.

        cov3D_precomp = None
        if self.pipe.compute_cov3D_python:
            cov3D_precomp = self.gaussian_model.get_covariance(scaling_modifier)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        colors_precomp = None
        if override_color is None:
            if self.pipe.convert_SHs_python:
                shs_view = self.gaussian_model.get_features.transpose(1, 2).view(-1, 3, (self.gaussian_model.max_sh_degree + 1) ** 2)
                dir_pp = (self.gaussian_model.get_xyz - viewpoint_camera.camera_center.repeat(self.gaussian_model.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.gaussian_model.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            colors_precomp = override_color


        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_rgb, radii = rasterizer_rgb(
            means3D=self.means3D[is_in_box][::sparsity],
            means2D=self.means2D[is_in_box][::sparsity],
            shs=self.shs[is_in_box][::sparsity],
            colors_precomp=colors_precomp,
            opacities=self.disk_kernel(self.opacity[is_in_box][::sparsity]) if show_disk else self.opacity[is_in_box][::sparsity],
            scales = scaling_modifier * (torch.full(self.scales.shape, point_size*0.1).to(self.scales.device)[is_in_box][::sparsity] if show_ptc else self.scales[is_in_box][::sparsity]),
            rotations = self.rotations[is_in_box][::sparsity],
            cov3D_precomp=cov3D_precomp)

        rendered_d, _ = rasterizer_depth(
            means3D=self.means3D[is_in_box][::sparsity],
            means2D=self.means2D[is_in_box][::sparsity],
            shs=self.shs[is_in_box][::sparsity],
            colors_precomp=colors_precomp,
            opacities=self.disk_kernel(self.opacity[is_in_box][::sparsity]) if show_disk else self.opacity[is_in_box][::sparsity],
            scales=scaling_modifier * (torch.full(self.scales.shape, point_size * 0.1).to(self.scales.device)[is_in_box][::sparsity] if show_ptc else self.scales[is_in_box][::sparsity]),
            rotations=self.rotations[is_in_box][::sparsity],
            cov3D_precomp=cov3D_precomp)

        # rendered_image, radii, allmap = rasterizer(
        #     means3D = self.means3D[is_in_box][::sparsity],
        #     means2D = self.means2D[is_in_box][::sparsity],
        #     shs = self.shs[is_in_box][::sparsity],
        #     colors_precomp = None,
        #     opacities = self.disk_kernel(self.opacity[is_in_box][::sparsity]) if show_disk else self.opacity[is_in_box][::sparsity],
        #     scales = scaling_modifier * (torch.full(self.scales.shape, point_size*0.1).to(self.scales.device)[is_in_box][::sparsity] if show_ptc else self.scales[is_in_box][::sparsity]),
        #     rotations = self.rotations[is_in_box][::sparsity],
        #     cov3D_precomp = None
        # )

        rets ={'render': rendered_rgb,
               'depth': rendered_d,
                'viewspace_points': self.means2D,
                "visibility_filter" : radii > 0,
                 "radii": radii
        }
        return rets

    def get_outputs(self, 
                    camera,
                    active_sh_degree,
                    override_color=None,
                    valid_range: tuple=None, 
                    split: bool=False, 
                    slider: float=0.5,
                    show_ptc: bool=False,
                    show_disk: bool=False,
                    point_size: float=0.01,
                    scaling_modifier: float=1., 
                    sparsity: int=1,
                    render_type: str="render",
                    render_type1: str="render", 
                    render_type2: str="render",
                    ):
        def get_result(results, type):
            if type in results.keys():
                return results[type]
            elif type == 'curvature':
                return self.color_map(gradient_map(results['surf_normal']))
            elif type == 'edge':
                return self.color_map(gradient_map(results['render']))
            elif type == 'depth':
                return results['depth']
            else:
                # handle exception as RGB render
                return results['render']

        results = self.render_viewer(camera, 
                                    active_sh_degree,
                                    override_color=override_color,
                                    scaling_modifier=scaling_modifier,
                                    bg_color=self.background_color,
                                    sparsity = sparsity,
                                    valid_range = valid_range,
                                    show_ptc = show_ptc, 
                                    show_disk = show_disk,
                                    point_size = point_size, 
                                    )
        if not split: 
            return get_result(results, render_type)
        else:
            result = torch.zeros_like(results['render'])
            _, _, render_h = result.shape
            slider_pos = int(render_h * slider)
            result[:, :, :slider_pos] = get_result(results, render_type1)[:, :, :slider_pos]
            result[:, :, slider_pos:] = get_result(results, render_type2)[:, :, slider_pos:]
            result[:, :, slider_pos] = torch.ones_like(result[:, :, slider_pos])

            return result