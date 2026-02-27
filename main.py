from camera import MechEyeCamera
from pose_est_cal_auto import run_pose_estimation



def main() -> int:
	cam = MechEyeCamera()
	cam.connect(None)
        
	try:
		run_pose_estimation(
			cam,
			# mesh_file="./model_ex2.obj",
			mesh_scale=0.001,
			fx=1805.0593503784967,
			fy=1805.167977554717,
			cx=649.2095330975123,
			cy=505.534917088949,
			out_width=640,
			out_height=512,
			depth_scale=0.001,
			window_name="MakinaRocks smart welding automation system",
			est_refine_iter=5,
			# on_pose=_on_pose,
			return_on_estimate=False,
		)
		
	finally:
		cam.disconnect()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

