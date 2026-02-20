import argparse
import os
import sys
import time
import warnings

import cv2
import numpy as np

from camera import MechEyeCamera


def _install_roi_mouse_callback(window_name: str, state: dict) -> None:
	"""Enable click-drag ROI rectangle selection on an OpenCV window.

	State keys written:
	- drawing: bool
	- x0,y0,x1,y1: int
	- has_roi: bool
	"""

	def _cb(event, x, y, flags, userdata):
		st = userdata
		# If the display is a composite (e.g., RGB | Depth), restrict ROI to the left RGB panel.
		img_w = int(st.get("img_w", 0) or 0)
		img_h = int(st.get("img_h", 0) or 0)
		if img_w > 0:
			# Ignore new drags that start outside RGB panel.
			if event == cv2.EVENT_LBUTTONDOWN and int(x) >= img_w:
				return
			# While drawing, clamp x into RGB panel so rectangles don't spill into the depth panel.
			if st.get("drawing", False):
				x = max(0, min(int(x), img_w - 1))
		if img_h > 0 and st.get("drawing", False):
			y = max(0, min(int(y), img_h - 1))

		if event == cv2.EVENT_LBUTTONDOWN:
			st["drawing"] = True
			st["x0"], st["y0"] = int(x), int(y)
			st["x1"], st["y1"] = int(x), int(y)
			st["has_roi"] = False
		elif event == cv2.EVENT_MOUSEMOVE and st.get("drawing", False):
			st["x1"], st["y1"] = int(x), int(y)
		elif event == cv2.EVENT_LBUTTONUP and st.get("drawing", False):
			st["drawing"] = False
			st["x1"], st["y1"] = int(x), int(y)
			# finalize
			x0, y0, x1, y1 = st.get("x0", 0), st.get("y0", 0), st.get("x1", 0), st.get("y1", 0)
			xa, xb = sorted([int(x0), int(x1)])
			ya, yb = sorted([int(y0), int(y1)])
			# Require a minimal size
			st["has_roi"] = (xb - xa) >= 5 and (yb - ya) >= 5
			st["x0"], st["y0"], st["x1"], st["y1"] = xa, ya, xb, yb

	cv2.setMouseCallback(window_name, _cb, state)


def _roi_to_mask(shape_hw: tuple[int, int], roi: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
	"""Create (mask_img_uint8, mask_bool) from ROI rectangle."""
	h, w = int(shape_hw[0]), int(shape_hw[1])
	x0, y0, x1, y1 = roi
	x0 = max(0, min(w - 1, int(x0)))
	x1 = max(0, min(w, int(x1)))
	y0 = max(0, min(h - 1, int(y0)))
	y1 = max(0, min(h, int(y1)))
	mask = np.zeros((h, w), dtype=np.uint8)
	if x1 > x0 and y1 > y0:
		mask[y0:y1, x0:x1] = 255
	mask_bool = mask > 0
	return mask, mask_bool


def _normalize_depth_for_view(depth_m: np.ndarray) -> np.ndarray:
	"""Convert float32 depth map (meters) to an 8-bit visualization."""

	if depth_m is None or depth_m.size == 0:
		return np.zeros((1, 1), dtype=np.uint8)

	d = depth_m.copy()
	invalid = ~np.isfinite(d) | (d <= 0)
	if np.all(invalid):
		return np.zeros(d.shape, dtype=np.uint8)

	d[invalid] = np.nan
	d_min = np.nanpercentile(d, 5)
	d_max = np.nanpercentile(d, 95)
	if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
		d_min, d_max = np.nanmin(d), np.nanmax(d)
		if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
			return np.zeros(d.shape, dtype=np.uint8)

	d = np.clip(d, d_min, d_max)
	d = (255.0 * (d - d_min) / (d_max - d_min)).astype(np.uint8)
	d[invalid] = 0
	return d


def _overlay_depth_edges(vis_bgr: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
	"""Overlay depth edges on top of BGR image for quick alignment sanity check."""
	depth_u8 = _normalize_depth_for_view(depth_m)
	if depth_u8.size == 0:
		return vis_bgr
	edges = cv2.Canny(depth_u8, 50, 150)
	if edges.ndim != 2:
		return vis_bgr
	overlay = np.zeros_like(vis_bgr)
	overlay[edges > 0] = (0, 0, 255)  # red edges
	return cv2.addWeighted(vis_bgr, 1.0, overlay, 0.7, 0)


def _load_K(args: argparse.Namespace, width: int, height: int) -> np.ndarray:
	if args.K_file:
		K = np.load(args.K_file)
		K = np.asarray(K, dtype=np.float32)
		if K.shape != (3, 3):
			raise ValueError(f"K_file must contain a 3x3 matrix, got {K.shape}")
		return K

	if args.fx > 0 and args.fy > 0:
		cx = args.cx if args.cx >= 0 else (width - 1) / 2.0
		cy = args.cy if args.cy >= 0 else (height - 1) / 2.0
		return np.array([[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

	# Fallback guess (will be inaccurate; user should provide calibration).
	fx = max(width, height) * 1.2
	fy = fx
	cx = (width - 1) / 2.0
	cy = (height - 1) / 2.0
	print("[WARN] No intrinsics provided; using a rough guessed K. Pass --K_file or --fx/--fy/--cx/--cy for accuracy.")
	return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
	K2 = np.asarray(K, dtype=np.float32).copy()
	K2[0, 0] *= float(sx)
	K2[1, 1] *= float(sy)
	K2[0, 2] *= float(sx)
	K2[1, 2] *= float(sy)
	return K2


def _load_mask(mask_file: str | None, width: int, height: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
	if not mask_file:
		return None, None
	mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
	if mask_img is None:
		raise FileNotFoundError(f"mask file not found: {mask_file}")
	if mask_img.shape[:2] != (height, width):
		mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
	mask_bool = mask_img > 0
	return mask_img, mask_bool


def _setup_foundationpose(args: argparse.Namespace):
	code_dir = os.path.dirname(os.path.realpath(__file__))
	object_pose_dir = os.path.join(code_dir, "object_pose")
	if object_pose_dir not in sys.path:
		sys.path.insert(0, object_pose_dir)

	warnings.filterwarnings("ignore")
	import logging

	logging.disable(logging.CRITICAL)

	# Imported after sys.path setup.
	from estimater import (
		FoundationPose,
		PoseRefinePredictor,
		ScorePredictor,
		dr,
		draw_posed_3d_box,
		draw_xyz_axis,
		set_seed,
		trimesh,
	)

	set_seed(0)

	mesh = trimesh.load(args.mesh_file)
	if getattr(args, "mesh_scale", 1.0) != 1.0:
		mesh.apply_scale(float(args.mesh_scale))
	mesh.vertices = mesh.vertices.astype(np.float32)
	# Ensure normals exist and dtype is consistent.
	try:
		normals = mesh.vertex_normals
		if normals is None or len(normals) != len(mesh.vertices):
			# Accessing vertex_normals triggers trimesh to compute them for meshes.
			normals = mesh.vertex_normals
		mesh.vertex_normals = np.asarray(normals, dtype=np.float32)
	except Exception:
		# If normals computation fails, fall back to zeros (pose quality may degrade).
		mesh.vertex_normals = np.zeros_like(mesh.vertices, dtype=np.float32)

	# If your OBJ is in millimeters, uncomment scaling in your own workflow.
	# mesh.apply_scale(0.001)

	to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
	extents = np.asarray(extents, dtype=np.float32)
	diam = float(np.linalg.norm(extents))
	print(f"[mesh] file={args.mesh_file}")
	print(f"[mesh] scale={float(getattr(args, 'mesh_scale', 1.0))}")
	print(f"[mesh] extents(maybe meters)={extents.tolist()}  diameter={diam:.6f}")
	if diam < 0.005:
		print("[mesh][WARN] Mesh seems very small (<5mm). If your mesh is in mm, try --mesh_scale 0.001")
	elif diam > 5.0:
		print("[mesh][WARN] Mesh seems very large (>5m). If your mesh is in mm but treated as m, try --mesh_scale 0.001")
	bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
	scorer = ScorePredictor()
	refiner = PoseRefinePredictor()
	glctx = dr.RasterizeCudaContext()

	debug_dir = os.path.join(code_dir, "debug_live_mechmind")
	os.makedirs(debug_dir, exist_ok=True)

	est = FoundationPose(
		model_pts=mesh.vertices,
		model_normals=mesh.vertex_normals,
		mesh=mesh,
		scorer=scorer,
		refiner=refiner,
		debug_dir=debug_dir,
		debug=0,
		glctx=glctx,
	)

	return est, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis


def main() -> int:
	parser = argparse.ArgumentParser(description="Live Mech-Eye capture / FoundationPose (ESC/q to quit)")
	parser.add_argument("--ip", type=str, default=None, help="Camera IP address (optional)")
	parser.add_argument(
		"--task",
		type=str,
		default="pose",
		choices=["viewer", "pose"],
		help="viewer: show RGB/Depth only, pose: FoundationPose tracking (mask-based init)",
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="rgbd",
		choices=["rgb", "rgbd"],
		help="rgb: 2D only, rgbd: 2D+3D (depth map)",
	)
	parser.add_argument("--window", type=str, default="Mech-Eye", help="OpenCV window title")
	parser.add_argument("--depth_window", type=str, default="Mech-Eye Depth", help="Depth window title")
	parser.add_argument("--max_fps", type=float, default=0.0, help="Limit FPS (0 = no limit)")
	parser.add_argument(
		"--debug_align",
		action="store_true",
		help="Overlay depth edges on RGB to visually verify RGB/Depth pixel alignment",
	)
	parser.add_argument(
		"--force_depth_resize_to_rgb",
		action="store_true",
		help="If depth shape differs from RGB, resize depth to match RGB (NEAREST). Use only if your camera provides unregistered depth.",
	)

	# Post-capture resize (display + pose processing). If 0, keep native size.
	# We default to 640x512 to ensure RGB/Depth inputs match this resolution.
	parser.add_argument("--out_width", type=int, default=640, help="Output width (resize after capture)")
	parser.add_argument("--out_height", type=int, default=512, help="Output height (resize after capture)")

	# FoundationPose args
	code_dir = os.path.dirname(os.path.realpath(__file__))
	object_pose_dir = os.path.join(code_dir, "object_pose")
	parser.add_argument(
		"--mesh_file",
		type=str,
		default=os.path.join(object_pose_dir, "obj_images", "mesh", "model.obj"),
		help="OBJ mesh path",
	)
	parser.add_argument(
		"--mesh_scale",
		type=float,
		default=0.001,
		help="Scale factor applied to mesh vertices before pose (use 0.001 for mm->m)",
	)
	parser.add_argument(
		"--mask_file",
		type=str,
		default=os.path.join(object_pose_dir, "obj_images", "masks", "000000.png"),
		help="Mask PNG path used for 's' initialization",
	)
	parser.add_argument("--est_refine_iter", type=int, default=5)
	parser.add_argument("--track_refine_iter", type=int, default=2)

	# Intrinsics: prefer --K_file (3x3 npy) or override fx/fy/cx/cy.
	# Default values provided by user calibration.
	parser.add_argument("--K_file", type=str, default=None, help="Path to 3x3 intrinsics matrix .npy")
	parser.add_argument(
		"--K_for_resized",
		action="store_true",
		help="Assume provided K already matches the resized resolution (skip auto scaling)",
	)
	parser.add_argument("--fx", type=float, default=1805.0593503784967)
	parser.add_argument("--fy", type=float, default=1805.167977554717)
	parser.add_argument("--cx", type=float, default=649.2095330975123)
	parser.add_argument("--cy", type=float, default=505.534917088949)

	# Depth conversion: depth_meters = depth_raw_float * depth_scale
	parser.add_argument(
		"--depth_scale",
		type=float,
		default=0.001,
		help="Multiply depth map by this to get meters (mm->m: 0.001, already meters: 1.0)",
	)
	args = parser.parse_args()

	cam = MechEyeCamera()
	cam.connect(args.ip)

	cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
	roi_state = {"drawing": False, "has_roi": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}
	_install_roi_mouse_callback(args.window, roi_state)
	# Depth is shown inside the main window (single-window view).

	est = None
	to_origin = None
	bbox = None
	draw_posed_3d_box = None
	draw_xyz_axis = None

	if args.task == "pose":
		est, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis = _setup_foundationpose(args)
		print("s: start (mask init), r: reset, q/ESC: quit")
	else:
		print("ESC/q: quit")

	pose = None
	initialized = False
	mask_img = None
	mask_bool = None
	K = None
	printed_K = False

	last_time = time.perf_counter()
	frame_count = 0
	fps = 0.0

	try:
		while True:
			t0 = time.perf_counter()

			if args.mode == "rgb":
				rgb = cam.capture_color()
				depth = None
			else:
				rgb, depth, _ = cam.capture_textured_point_cloud()

			cap_h, cap_w = rgb.shape[:2]

			# Ensure float32 depth.
			if depth is not None and depth.dtype != np.float32:
				depth = depth.astype(np.float32)

			# If RGB/Depth resolutions differ, FoundationPose inputs are not pixel-aligned.
			# (We can optionally force-resize depth, but true alignment requires registration.)
			if depth is not None and depth.shape[:2] != rgb.shape[:2]:
				if args.force_depth_resize_to_rgb:
					depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
				else:
					# Keep as-is; we will warn on-screen.
					pass

			# Resize after capture (affects BOTH display and pose processing).
			if args.out_width <= 0 or args.out_height <= 0:
				raise ValueError("out_width/out_height must be positive")
			if (rgb.shape[1] != args.out_width) or (rgb.shape[0] != args.out_height):
				rgb = cv2.resize(
					rgb,
					(args.out_width, args.out_height),
					interpolation=cv2.INTER_AREA if rgb.shape[0] > args.out_height else cv2.INTER_LINEAR,
				)
				if depth is not None:
					depth = cv2.resize(depth, (args.out_width, args.out_height), interpolation=cv2.INTER_NEAREST)

			height, width = rgb.shape[:2]

			# Lazy-load K + mask after we know final processing resolution.
			if args.task == "pose" and K is None:
				K_native = _load_K(args, cap_w, cap_h)
				if not args.K_for_resized:
					sx = float(width) / float(cap_w)
					sy = float(height) / float(cap_h)
					K = _scale_K(K_native, sx=sx, sy=sy)
					print(f"[K] Auto-scaled intrinsics: {cap_w}x{cap_h} -> {width}x{height} (sx={sx:.4f}, sy={sy:.4f})")
				else:
					K = K_native
				mask_img, mask_bool = _load_mask(args.mask_file, width, height)
				printed_K = False

			if args.task == "pose" and (K is not None) and not printed_K:
				print(f"[K] Using K=\n{K}")
				printed_K = True

			# FPS
			frame_count += 1
			now = time.perf_counter()
			dt = now - last_time
			if dt >= 0.5:
				fps = frame_count / dt
				frame_count = 0
				last_time = now

			# Prepare inputs
			color_rgb = rgb  # already RGB
			depth_m = None
			if depth is not None:
				depth_m = depth.astype(np.float32) * float(args.depth_scale)

			# Visualization base
			vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
			if depth is not None and depth.shape[:2] != rgb.shape[:2]:
				cv2.putText(
					vis_bgr,
					f"WARN: depth {depth.shape[1]}x{depth.shape[0]} != rgb {width}x{height}",
					(10, 90),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 0, 255),
					2,
					cv2.LINE_AA,
				)

			# Draw ROI rectangle (drag with left mouse)
			if roi_state.get("drawing", False) or roi_state.get("has_roi", False):
				x0, y0, x1, y1 = roi_state.get("x0", 0), roi_state.get("y0", 0), roi_state.get("x1", 0), roi_state.get("y1", 0)
				# Clamp to current image
				x0 = max(0, min(width - 1, int(x0)))
				x1 = max(0, min(width - 1, int(x1)))
				y0 = max(0, min(height - 1, int(y0)))
				y1 = max(0, min(height - 1, int(y1)))
				cv2.rectangle(vis_bgr, (x0, y0), (x1, y1), (0, 255, 255), 2)

			if args.task == "pose":
				if not initialized:
					cv2.putText(
						vis_bgr,
						"Drag ROI with mouse, then press 's' (or use mask file)",
						(10, 30),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.7,
						(0, 255, 255),
						2,
					)
				else:
					if pose is not None and depth_m is not None:
						pose = est.track_one(
							rgb=color_rgb,
							depth=depth_m,
							K=K,
							iteration=int(args.track_refine_iter),
						)
						center_pose = pose @ np.linalg.inv(to_origin)

						vis_rgb = draw_posed_3d_box(K, img=color_rgb, ob_in_cam=center_pose, bbox=bbox)
						vis_rgb = draw_xyz_axis(
							color_rgb,
							ob_in_cam=center_pose,
							scale=0.1,
							K=K,
							thickness=3,
							transparency=0,
							is_input_rgb=True,
						)
						vis_bgr = vis_rgb[..., ::-1]
						vis_bgr = np.ascontiguousarray(vis_bgr)
						if vis_bgr.dtype != np.uint8:
							vis_bgr = np.clip(vis_bgr, 0, 255).astype(np.uint8)

			cv2.putText(
				vis_bgr,
				f"FPS: {fps:.1f}  {width}x{height}",
				(10, 60),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 0),
				2,
				cv2.LINE_AA,
			)

			# Keep ROI interaction bounded to the left (RGB) panel.
			roi_state["img_w"] = int(width)
			roi_state["img_h"] = int(height)

			# Alignment debug overlay (after drawing pose/ROI/text)
			if args.debug_align and depth_m is not None and depth_m.shape[:2] == rgb.shape[:2]:
				vis_bgr = _overlay_depth_edges(vis_bgr, depth_m)

			# Single-window composite view: [RGB inference | Depth]
			if depth_m is not None:
				depth_u8 = _normalize_depth_for_view(depth_m)
				depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
				# Ensure depth panel matches RGB panel size.
				if depth_color.shape[:2] != (height, width):
					depth_color = cv2.resize(depth_color, (width, height), interpolation=cv2.INTER_NEAREST)
				cv2.putText(
					depth_color,
					"Depth",
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.9,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)
				combo = np.hstack([vis_bgr, depth_color])
				# Optional divider line
				cv2.line(combo, (width, 0), (width, height - 1), (40, 40, 40), 2)
				cv2.imshow(args.window, combo)
			else:
				cv2.imshow(args.window, vis_bgr)

			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break
			if args.task == "pose":
				if key == ord("r"):
					initialized = False
					pose = None
					print("\nReset")
				elif key == ord("c"):
					roi_state["drawing"] = False
					roi_state["has_roi"] = False
					print("\nCleared ROI")
				elif key == ord("s"):
					if depth_m is None:
						print("\n[WARN] No depth available (mode=rgb). Use --mode rgbd for pose.")
					else:
						mask_bool_to_use = mask_bool
						if roi_state.get("has_roi", False):
							x0, y0, x1, y1 = roi_state.get("x0", 0), roi_state.get("y0", 0), roi_state.get("x1", 0), roi_state.get("y1", 0)
							mask_img_roi, mask_bool_roi = _roi_to_mask((height, width), (x0, y0, x1, y1))
							mask_bool_to_use = mask_bool_roi
							print(f"\nUsing ROI mask: ({x0},{y0})-({x1},{y1})")
						elif mask_bool is None:
							print("\n[WARN] No ROI selected and mask file not loaded.")
						pose = est.register(
							K=K,
							rgb=color_rgb,
							depth=depth_m,
							ob_mask=mask_bool_to_use,
							iteration=int(args.est_refine_iter),
						)
						initialized = True
						print("\nInitialized!")

			if args.max_fps and args.max_fps > 0:
				elapsed = time.perf_counter() - t0
				sleep_s = max(0.0, (1.0 / args.max_fps) - elapsed)
				if sleep_s > 0:
					time.sleep(sleep_s)
	finally:
		cam.disconnect()
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

