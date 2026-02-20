#!/usr/bin/env python3
"""
저장된 RGB/Depth/Mask로 Pose Estimation 테스트
"""

from estimater import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/obj_images/mesh/model.obj')
    parser.add_argument('--rgb_file', type=str, default=f'{code_dir}/obj_images/rgb/000000.png')
    parser.add_argument('--depth_file', type=str, default=f'{code_dir}/obj_images/depth/000000.png')
    parser.add_argument('--mask_file', type=str, default=f'{code_dir}/obj_images/masks/000000.png')
    parser.add_argument('--cam_K', type=str, default=f'{code_dir}/obj_images/cam_K.txt')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # Mesh 로드
    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    # FoundationPose 초기화
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=f'{code_dir}/debug_static', debug=0, glctx=glctx)
    logging.info("FoundationPose initialized")

    # 카메라 K 로드
    K = np.loadtxt(args.cam_K).reshape(3, 3)
    logging.info(f"K:\n{K}")

    # RGB 로드 (BGR -> RGB)
    rgb_bgr = cv2.imread(args.rgb_file)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    logging.info(f"RGB shape: {rgb.shape}")

    # Depth 로드 (16-bit PNG, mm -> m)
    depth_mm = cv2.imread(args.depth_file, cv2.IMREAD_UNCHANGED)
    depth = depth_mm.astype(np.float32) / 1000.0
    logging.info(f"Depth shape: {depth.shape}, range: {depth.min():.3f} ~ {depth.max():.3f} m")

    # Mask 로드
    mask_img = cv2.imread(args.mask_file, cv2.IMREAD_GRAYSCALE)
    mask = (mask_img > 0).astype(bool)
    logging.info(f"Mask shape: {mask.shape}, pixels: {mask.sum()}")

    # Pose 추정
    logging.info("Running pose estimation...")
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    logging.info(f"Pose:\n{pose}")

    # 시각화
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)

    # 결과 저장 및 표시
    vis_bgr = vis[..., ::-1]
    cv2.imwrite(f'{code_dir}/debug_static/result.png', vis_bgr)
    logging.info(f"Result saved to {code_dir}/debug_static/result.png")

    cv2.imshow('Pose Result', vis_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
