#
# Copyright (C) 2023, Inria
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr


# ============================================================
# === SIMPLE INFERENCE-ONLY OPACITY CULLING ==================
# ============================================================

def screen_space_radius_cull(gaussians, view, min_radius_px=1.0):
    """
    Cull Gaussians that project to < min_radius_px pixels.
    """
    xyz = gaussians.get_xyz                      # (N,3)
    scales = gaussians.get_scaling.max(dim=1).values  # approximate radius

    ones = torch.ones((xyz.shape[0], 1), device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=1)

    clip = (view.full_proj_transform @ xyz_h.T).T
    z = clip[:, 2].abs().clamp_min(1e-4)

    # approximate focal length from FoV
    focal = 0.5 * view.image_width / torch.tan(torch.tensor(view.FoVx * 0.5, device=xyz.device))

    projected_radius = focal * scales / z
    return projected_radius >= min_radius_px

def distance_lod_cull(
    gaussians,
    view,
    far=6.0,
    min_opacity=0.02,
    min_radius_px=0.5
):
    xyz = gaussians.get_xyz
    cam = view.camera_center
    opacity = gaussians.get_opacity.squeeze()
    scales = gaussians.get_scaling.max(dim=1).values

    dist = torch.norm(xyz - cam, dim=1)

    # project radius
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=1)
    clip = (view.full_proj_transform @ xyz_h.T).T
    z = clip[:, 2].abs().clamp_min(1e-4)

    focal = 0.5 * view.image_width / torch.tan(
        torch.tensor(view.FoVx * 0.5, device=xyz.device)
    )
    projected_radius = focal * scales / z

    keep = torch.ones_like(dist, dtype=torch.bool)

    far_mask = dist > far
    drop_far = (
        (opacity < min_opacity) |
        (projected_radius < min_radius_px)
    )

    keep[far_mask & drop_far] = False
    return keep

def frustum_cull_gaussians(gaussians, view, margin=1.2):
    """
    Conservative frustum culling with Gaussian extent.
    margin > 1 makes it safer.
    """
    xyz = gaussians.get_xyz
    scales = gaussians.get_scaling.max(dim=1).values

    ones = torch.ones((xyz.shape[0], 1), device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=1)

    clip = (view.full_proj_transform @ xyz_h.T).T
    w = clip[:, 3].abs().clamp_min(1e-4)

    # approximate projected radius in clip space
    radius = scales * margin * w

    inside = (
        (clip[:, 0] >= -w - radius) & (clip[:, 0] <= w + radius) &
        (clip[:, 1] >= -w - radius) & (clip[:, 1] <= w + radius) &
        (clip[:, 2] >= -radius) & (clip[:, 2] <= w + radius)
    )
    return inside


def opacity_cull_gaussians(gaussians, min_opacity):
    return gaussians.get_opacity.squeeze() > min_opacity


def make_pruned_gaussians(src: GaussianModel, mask: torch.Tensor):
    g = GaussianModel(src.max_sh_degree)
    g.active_sh_degree = src.active_sh_degree

    g._xyz = src._xyz[mask]
    g._features_dc = src._features_dc[mask]
    g._features_rest = src._features_rest[mask]
    g._opacity = src._opacity[mask]
    g._scaling = src._scaling[mask]
    g._rotation = src._rotation[mask]

    # Copy exposure ONLY if it exists
    if hasattr(src, "_exposure"):
        g._exposure = src._exposure
        g.exposure_mapping = src.exposure_mapping
        g.pretrained_exposures = src.pretrained_exposures

    return g


# ============================================================
# === BASELINE RENDER (UNCHANGED) ============================
# ============================================================

def render_set(model_path, name, iteration, views, gaussians,
               pipeline, background, train_test_exp, separate_sh):

    render_path = os.path.join(model_path, name, f"baseline_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"baseline_{iteration}", "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    psnrs, fpss, times = [], [], []

    print("Warm up (baseline)")
    for _ in range(10):
        _ = render(views[0], gaussians, pipeline, background,
                   use_trained_exp=train_test_exp, separate_sh=separate_sh)

    for idx, view in enumerate(tqdm(views, desc="Rendering baseline")):
        pack = render(view, gaussians, pipeline, background,
                      use_trained_exp=train_test_exp, separate_sh=separate_sh)

        rendering = pack["render"]
        render_time = pack["render_time"]
        gt = view.original_image[0:3]

        psnrs.append(psnr(rendering, gt).mean())
        fpss.append(1.0 / render_time)
        times.append(render_time)

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

    print("psnrs=", torch.tensor(psnrs).mean())
    print("fps=", torch.tensor(fpss).mean())
    print("time=", torch.tensor(times).mean())


# ============================================================
# === OPACITY-CULLED RENDER ================================
# ============================================================

def culling_render(model_path, name, iteration, views, gaussians,
                   pipeline, background, train_test_exp,
                   separate_sh, min_opacity, frustum= False, radius=False, lod=False, all_cullings=False):

    render_path = os.path.join(model_path, name, f"opacity_cull_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"opacity_cull_{iteration}", "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    mask_opacity = opacity_cull_gaussians(gaussians, min_opacity)
    mask_frustum = frustum_cull_gaussians(gaussians, views[0])
    mask_radius  = screen_space_radius_cull(gaussians, views[0], min_radius_px=1.0)
    mask_lod     = distance_lod_cull(gaussians, views[0])


    # Always start from opacity
    mask = mask_opacity.clone()

    # Treat --use_all_culling as enabling everything
    use_radius  = radius  or all_cullings
    use_frustum = frustum or all_cullings
    use_lod     = lod     or all_cullings

    # Apply progressively (importance order)
    if use_radius:
        mask &= mask_radius

    if use_frustum:
        mask &= mask_frustum

    if use_lod:
        mask &= mask_lod

    
    render_gaussians = make_pruned_gaussians(gaussians, mask)
    print(f"[CULL] kept {mask.sum().item()} / {mask.numel()}")

    #     print(f"[CULL] kept {mask.sum().item()} / {mask.numel()}")
    # else:
    #     render_gaussians = make_pruned_gaussians(gaussians, mask)
    #     print(f"[Opacity culling] kept {mask.sum().item()} / {mask.numel()} gaussians")

    psnrs, fpss, times = [], [], []

    print("Warm up (culled)")
    for _ in range(10):
        _ = render(views[0], render_gaussians, pipeline, background,
                   use_trained_exp=train_test_exp, separate_sh=separate_sh)

    for idx, view in enumerate(tqdm(views, desc="Rendering opacity-culled")):
        pack = render(view, render_gaussians, pipeline, background,
                      use_trained_exp=train_test_exp, separate_sh=separate_sh)

        rendering = pack["render"]
        render_time = pack["render_time"]
        gt = view.original_image[0:3]

        psnrs.append(psnr(rendering, gt).mean())
        fpss.append(1.0 / render_time)
        times.append(render_time)

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

    print("psnrs=", torch.tensor(psnrs).mean())
    print("fps=", torch.tensor(fpss).mean())
    print("time=", torch.tensor(times).mean())


# ============================================================
# === ENTRYPOINT ============================================
# ============================================================

def render_sets(dataset, iteration, pipeline, skip_train, skip_test,
                separate_sh, use_opacity_culling, min_opacity,
                frustum, radius, lod, all_cullings):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        def run(name, views):
            if use_opacity_culling:
                culling_render(
                    dataset.model_path, name, scene.loaded_iter,
                    views, gaussians, pipeline, background,
                    dataset.train_test_exp, separate_sh,
                    min_opacity,
                    frustum=frustum,
                    radius=radius,
                    lod=lod,
                    all_cullings=all_cullings
                )

            else:
                render_set(dataset.model_path, name, scene.loaded_iter,
                           views, gaussians, pipeline, background,
                           dataset.train_test_exp, separate_sh)

        if not skip_train:
            run("train", scene.getTrainCameras())
        if not skip_test:
            run("test", scene.getTestCameras())


# ============================================================
# === MAIN ===================================================
# ============================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--use_opacity_culling", action="store_true")
    parser.add_argument("--min_opacity", type=float, default=0.01)
    parser.add_argument("--use_frustum_culling", action="store_true")
    parser.add_argument("--use_radius_culling", action="store_true")
    parser.add_argument("--use_lod_culling", action="store_true")
    parser.add_argument("--use_all_culling", action="store_true")


    args = get_combined_args(parser)
    args.eval = True
    args.depths = ""
    args.train_test_exp = False

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        False,
        args.use_opacity_culling,
        args.min_opacity,
        args.use_frustum_culling,
        args.use_radius_culling,
        args.use_lod_culling,
        args.use_all_culling,
    )
