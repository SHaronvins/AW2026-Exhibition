from ultralytics import YOLO



def main():
    # 🔥 SEGMENTATION 모델 사용
    model = YOLO("yolo11s-seg.pt")

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=1024,
        batch=8,

        # 🔥 증강 (현재 세팅 유지)
        degrees=15.0,
        fliplr=0.5,
        flipud=0.5,
        translate=0.04,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        rect=True,
        plots=False,
        lr0=0.001,
        lrf=0.01,

        # 🔥 이름 변경
        name="seg_train",
    )


if __name__ == "__main__":
    main()
