# -*- coding: utf-8 -*-  

"""
Created on 2021/4/14

@author: Ruoyu Chen
"""

import cv2
import dlib
import numpy as np

class Face_Align(object):
    def __init__(self,shape_predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

    def rect_to_tuple(self, rect):
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        return left, top, right, bottom

    def extract_eye(self, shape, eye_indices):
        points = map(lambda i: shape.part(i), eye_indices)
        return list(points)

    def extract_eye_center(self, shape, eye_indices):
        points = self.extract_eye(shape, eye_indices)
        xs = map(lambda p: p.x, points)
        ys = map(lambda p: p.y, points)
        return sum(xs) // 6, sum(ys) // 6

    def extract_left_eye_center(self, shape):
        return self.extract_eye_center(shape, self.LEFT_EYE_INDICES)

    def extract_right_eye_center(self, shape):
        return self.extract_eye_center(shape, self.RIGHT_EYE_INDICES)

    def angle_between_2_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def get_rotation_matrix(self, p1, p2):
        angle = self.angle_between_2_points(p1, p2)
        x1, y1 = p1
        x2, y2 = p2
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
        return M

    def crop_image(self, image, det):
        left, top, right, bottom = self.rect_to_tuple(det)
        return image[top:bottom, left:right]

    def __call__(self, image=None,image_path=None,save_path=None,only_one=True):
        '''
        Face alignment, can select input image variable or image path, when input
        image format that return alignment face image crop or image path as input
        will return None but save image to the save path.
        :image: Face image input
        :image_path: if image is None than can input image
        :save_path: path to save image
        :detector: detector = dlib.get_frontal_face_detector()
        :predictor: predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        '''
        if image is not None:
            # convert BGR format to Gray
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        elif image_path is not None:
            image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_path)
        
        height, width = image.shape[:2]

        # Dector face
        dets = self.detector(image_gray, 1)

        # i donate the i_th face detected in image
        crop_images = []
        for i, det in enumerate(dets):
            shape = self.predictor(image_gray, det)

            left_eye = self.extract_left_eye_center(shape)
            right_eye = self.extract_right_eye_center(shape)

            M = self.get_rotation_matrix(left_eye, right_eye)
            
            rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC)

            cropped = self.crop_image(rotated, det)

            if only_one == True:
                if save_path is not None:
                    cv2.imwrite(save_path, cropped)
                return cropped
            else:
                crop_images.append(cropped)
        return crop_images
    def real_fake(self,real,fake):
        image_gray = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
        height, width = real.shape[:2]

        # Dector face
        dets = self.detector(image_gray, 1)

        for i, det in enumerate(dets):
            shape = self.predictor(image_gray, det)

            left_eye = self.extract_left_eye_center(shape)
            right_eye = self.extract_right_eye_center(shape)

            M = self.get_rotation_matrix(left_eye, right_eye)
            
            rotated = cv2.warpAffine(real, M, (width, height), flags=cv2.INTER_CUBIC)
            rotated_f = cv2.warpAffine(fake, M, (width, height), flags=cv2.INTER_CUBIC)
            cropped = self.crop_image(rotated, det)
            cropped_f = self.crop_image(rotated_f, det)
        
        return cropped,cropped_f



if __name__ == "__main__":
    align = Face_Align("./face-alignment-dlib/shape_predictor_68_face_landmarks.dat")
    align(image_path="./face-alignment-dlib/123.jpg",save_path="test.jpg")