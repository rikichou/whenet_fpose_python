import cv2
import numpy as np

from . import whenet

class Pose():
    def __init__(self, model_path='./model/WHENet.h5'):
        self.model_path = model_path
        self.model = whenet.WHENet(snapshot=model_path)

    def get_face_image(self, img, face_rect):
        x_min, y_min, x_max, y_max = face_rect
        # enlarge the bbox to include more background margin
        y_min = max(0, y_min - abs(y_min - y_max) / 10)
        y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
        x_min = max(0, x_min - abs(x_min - x_max) / 5)
        x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
        x_max = min(x_max, img.shape[1])

        face_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        return face_img

    def preprocessing(self, face_image):
        img_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))
        img_rgb = np.expand_dims(img_rgb, axis=0)

        return img_rgb

    def __call__(self, image, face_rect):
        # get input face
        face_img = self.get_face_image(image, face_rect)

        # preprocessing
        input_image = self.preprocessing(face_img)

        # predict
        yaw, pitch, roll = self.model.get_angle(input_image)
        yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

        return yaw, pitch, roll