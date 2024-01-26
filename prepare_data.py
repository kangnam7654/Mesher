import argparse
from pathlib import Path
import cv2
from facenet_pytorch import MTCNN
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


landmarker_model_path = "./weights/face_landmarker.task"
detector_model_path = "./weights/blaze_face_short_range.tflite"


def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def get_face_info(numpy_image):
    # | Load Args |
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # | Set Options |
    landmark_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    detection_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=detector_model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    # | Load Image |
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    # | Inference |
    with FaceLandmarker.create_from_options(landmark_options) as landmarker:
        face_landmark_result = landmarker.detect(mp_image)

    with FaceDetector.create_from_options(detection_options) as detector:
        face_detection_result = detector.detect(mp_image)

    # | Parse to Boudning Box |
    bbox_result = face_detection_result.detections[0].bounding_box
    bbox_x = bbox_result.origin_x
    bbox_y = bbox_result.origin_y
    bbox_w = bbox_result.width
    bbox_h = bbox_result.height

    bbox = [bbox_x, bbox_y, bbox_w, bbox_h]

    # | Parse to Landmark |
    landmark = face_landmark_result.face_landmarks[0]

    return bbox, landmark


def main(args):
    root_dir = Path(args.root_dir)
    image_files = list(root_dir.rglob("*.jpg")) + list(root_dir.rglob("*.png"))

    for image_file in image_files:
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox, landmark = get_face_info(image)
        cropped = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :]
        
        # TODO 
        save_path = "test_crop.jpg"
        cv2.imwrite()


if __name__ == "__main__":
    image = cv2.imread("/Users/kangnam/dataset/ffhq/00000/00003.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    bbox, landamrk = get_face_info(numpy_image=image)
    cropped = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :]
