import sys
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads automatically the first time

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def main(source: str, target: str) -> None:
    video_info = sv.VideoInfo.from_video_path(source)
    frames = sv.get_video_frames_generator(source)

    with sv.VideoSink(target, video_info) as sink:
        for frame in frames:
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            labels = [
                f"{model.names[class_id]} {conf:.2f}"
                for class_id, conf in zip(detections.class_id, detections.confidence)
            ]

            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
            sink.write_frame(annotated)

    print(f"Saved to {target}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])