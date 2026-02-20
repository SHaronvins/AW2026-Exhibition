import sys
from mecheye.shared import *
from mecheye.area_scan_3d_camera import Camera, Frame2D, Frame3D, Frame2DAnd3D
import numpy as np


class MechEyeCamera:
    def __init__(self):
        self.cam = Camera()
        self.frame_all_2d_3d = Frame2DAnd3D()

    def connect(self, ip_address: str | None = None):
        """Connect to Mech-Eye camera.

        If `ip_address` is provided, try direct IP connection first.
        Otherwise, fall back to discovery and auto-select the first camera.
        """

        if ip_address:
            print(f"Connecting by IP: {ip_address}")
            status = self.cam.connect(ip_address)
            ok = status.is_ok() if hasattr(status, "is_ok") else status.isOK()
            if ok:
                print("Mech-Eye camera connected (by IP).")
                return
            raise RuntimeError(f"❌ Failed to connect to {ip_address}: {status}")

        camera_list = Camera.discover_cameras()
        if camera_list is None or len(camera_list) == 0:
            raise RuntimeError(
                "No Mech-Eye camera detected. "
                "(Try passing ip_address='192.168.23.203' if you know the camera IP.)"
            )

        print("Found cameras:")
        for i, info in enumerate(camera_list):
            print(f"  [{i}]  Model: {info.model}, IP: {info.ip_address}")

        chosen = camera_list[0]
        print(f"\n➡️ Auto-selecting camera 0: {chosen.model} @ {chosen.ip_address}")

        status = self.cam.connect(chosen)
        ok = status.is_ok() if hasattr(status, "is_ok") else status.isOK()
        if not ok:
            raise RuntimeError(f"Failed to connect: {status}")

        print("✅ Mech-Eye camera connected automatically.")

    def capture_color(self):
        frame2d = Frame2D()
        status = self.cam.capture_2d(frame2d)
        ok = status.is_ok() if hasattr(status, "is_ok") else status.isOK()
        if not ok:
            raise RuntimeError(f"Capture 2D failed: {status}")

        color = frame2d.get_color_image()
        if color is None or color.is_empty():
            raise RuntimeError("Color2DImage empty")

        w, h = color.width(), color.height()
        buf = color.data()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
        return img

    def capture_depth_and_pcd(self, ply_path="./PointCloud.ply"):

        show_error(self.cam.capture_2d_and_3d(self.frame_all_2d_3d))

        frame3d = self.frame_all_2d_3d.frame_3d()

        depth_img = frame3d.get_depth_map()
        if depth_img is None or depth_img.is_empty():
            raise RuntimeError("Depth Map empty")

        w, h = depth_img.width(), depth_img.height()
        depth = np.frombuffer(depth_img.data(), dtype=np.float32).reshape((h, w))

        show_error(frame3d.save_untextured_point_cloud(FileFormat_PLY, ply_path),
                   f"Saved untextured PLY → {ply_path}")

        return depth, ply_path

    def capture_textured_point_cloud(self, ply_path="TexturedPointCloud.ply"):

        # 2D+3D 캡처 (Frame2DAnd3D 사용)
        show_error(self.cam.capture_2d_and_3d(self.frame_all_2d_3d))

        # ---- 텍스처 PLY 저장 ----
        # intentionally disabled (do not write PLY)

        # ---- RGB ----
        frame2d = self.frame_all_2d_3d.frame_2d()
        color = frame2d.get_color_image()
        if color is None or color.is_empty():
            raise RuntimeError("Color2DImage empty")

        w, h = color.width(), color.height()
        img = np.frombuffer(color.data(), dtype=np.uint8).reshape((h, w, 3))

        # ---- Depth ----
        frame3d = self.frame_all_2d_3d.frame_3d()
        depth_img = frame3d.get_depth_map()
        if depth_img is None or depth_img.is_empty():
            raise RuntimeError("Depth Map empty")

        dw, dh = depth_img.width(), depth_img.height()
        depth = np.frombuffer(depth_img.data(), dtype=np.float32).reshape((dh, dw))

        return img, depth, ply_path

    def capture_all(self):
        return self.capture_textured_point_cloud("TexturedPointCloud.ply")

    def disconnect(self):
        self.cam.disconnect()
        print("🔌 Camera disconnected.")