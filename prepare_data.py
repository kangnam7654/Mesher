import cv2
from facenet_pytorch import MTCNN
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

landmarker_model_path = "/home/kangnam/project/Mesher/weights/face_landmarker.task"
detector_model_path = "/home/kangnam/project/Mesher/weights/blaze_face_short_range.tflite"

def get_face_landmark():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_model_path),
        running_mode=VisionRunningMode.IMAGE)

    with FaceLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # ...
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file('/home/kangnam/datasets/raw/ffhq/00000/00003.png')
        # Perform face landmarking on the provided single image.
        # The face landmarker must be created with the image mode.
        face_landmarker_result = landmarker.detect(mp_image)
    return face_landmarker_result

def get_face_detection():
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a face detector instance with the image mode:
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=detector_model_path),
        running_mode=VisionRunningMode.IMAGE)
    with FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image.create_from_file('/home/kangnam/datasets/raw/ffhq/00000/00003.png')
        face_detector_result = detector.detect(mp_image)
    return face_detector_result

if __name__ == "__main__":
    result = get_face_detection()