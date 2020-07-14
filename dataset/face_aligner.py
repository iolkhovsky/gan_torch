import dlib
import cv2
import numpy as np
from imutils import face_utils

DEFAULT_KPREGRESSOR_PATH = "/home/igor/github/my/GAN_faces/configs/shape_predictor_68_face_landmarks.dat"


class FaceAligner:

    def __init__(self, target_size=100, eye_level=0.3, eye_dist=0.4, kpregressor=DEFAULT_KPREGRESSOR_PATH, ):
        self.kepoint_regressor_path = kpregressor
        self.regressor = dlib.shape_predictor(self.kepoint_regressor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.out_target_size = target_size
        self.eye_level = eye_level
        self.eye_dist = eye_dist

        self.left_eye_ids = [36, 37, 38, 39, 40, 41]
        self.right_eye_ids = [42, 43, 44, 45, 46, 47]
        self.regressor_in_size = 500
        return

    @staticmethod
    def __distance_between_points(p1, p2):
        return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p1[1], 2))

    @staticmethod
    def __make_center(current, target=0.5):
        if current < target:
            output_offset = (target - current) / (1.0 - target)
            output_size = 1.0 + output_offset
        else:
            output_offset = 0
            output_size = current / target  # - 1.0
        return output_size, output_offset

    def process(self, image):
        image = cv2.resize(image, (self.regressor_in_size, self.regressor_in_size))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) < 1:
            return None

        key_points = self.regressor(gray, rects[0])
        key_points = face_utils.shape_to_np(key_points)
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])

        eye_l, eye_r = (0, 0), (0, 0)
        for idx, (x, y) in enumerate(key_points):
            if idx in self.left_eye_ids:
                eye_l = eye_l[0] + x, eye_l[1] + y
            if idx in self.right_eye_ids:
                eye_r = eye_r[0] + x, eye_r[1] + y
        eye_l = int(eye_l[0] / len(self.left_eye_ids)), int(eye_l[1] / len(self.left_eye_ids))
        eye_r = int(eye_r[0] / len(self.right_eye_ids)), int(eye_r[1] / len(self.right_eye_ids))
        center = int((eye_l[0] + eye_r[0]) / 2), int((eye_l[1] + eye_r[1]) / 2)
        eye_distance = self.__distance_between_points(eye_l, eye_r)

        buf_sz_x, buf_offset_x = self.__make_center(current=center[0] / image.shape[1], target=0.5)
        buf_sz_y, buf_offset_y = self.__make_center(current=center[1] / image.shape[0], target=self.eye_level)

        buf_sz_x = int(buf_sz_x * image.shape[1])
        buf_sz_y = int(buf_sz_y * image.shape[0])
        buf_offset_x = int(buf_offset_x * image.shape[1])
        buf_offset_y = int(buf_offset_y * image.shape[0])

        extended_size = max(buf_sz_x, buf_sz_y)
        eye_target_dist = self.eye_dist * extended_size

        buf = np.zeros(shape=(extended_size, extended_size, 3), dtype=np.uint8)
        buf[buf_offset_y:buf_offset_y + self.regressor_in_size, buf_offset_x:buf_offset_x + self.regressor_in_size, :] \
            = image.copy()

        angle = -1 * np.arctan((center[1] - eye_r[1]) / (eye_r[0] - center[0])) * 180. / np.pi
        scale = eye_target_dist / eye_distance
        center = int(0.5 * extended_size), int(self.eye_level * extended_size)
        transform_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(buf, transform_matrix, (extended_size, extended_size))

        return cv2.resize(rotated, (self.out_target_size, self.out_target_size))

    def __call__(self, *args, **kwargs):
        return self.process(args[0])

