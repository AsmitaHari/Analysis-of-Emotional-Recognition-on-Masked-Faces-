import os
import pathlib
import sys
import argparse
import numpy as np
import newFile
from PIL import Image, ImageFile
import glob
import pandas as pd
import torch
import math
from shutil import copyfile

__version__ = '0.3.0'

IMAGE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, "surgicalMask.png")
BLACK_IMAGE_PATH = os.path.join(IMAGE_DIR, 'black-mask.png')
BLUE_IMAGE_PATH = os.path.join(IMAGE_DIR, 'blue-mask.png')
RED_IMAGE_PATH = os.path.join(IMAGE_DIR, 'red-mask.png')


def cli(img,outFolder):

    #New folder\\
    pic_path = img
    if not os.path.exists(pic_path):
        print(f'Picture {pic_path} does not exist.')
        sys.exit(1)

    mask_path = DEFAULT_IMAGE_PATH

    FaceMasker(pic_path, mask_path,outFile=outFolder).mask()
    debug=0


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog',outFile=None):
        self.face_path = face_path
        self.mask_path = mask_path
        if outFile==None:
            self.outFile=face_path
        else:
            self.outFile=outFile
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None

    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            self._mask_face(face_landmark)

        if found_face:
            if self.show:
                self._face_img.show()

            # save
            self._save()
        else:
            print('Found no face.')

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        path_splits = os.path.splitext(self.face_path)

        outpath=self.outFile+"\\"+path_splits[0].split("\\")[-1]
        if not os.path.exists(self.outFile):
            os.makedirs(self.outFile)
        new_face_path = outpath + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        newFile.saveLandMarks(new_face_path,self.outFile, path_splits[0].split("\\")[-1])

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
        #pass the name of the folder you want to look at here
        print("test")

        list = []
        index = 1
        # for filepath in glob.glob('C:/RIT-Stuff/Topics in System/Project/extended-cohn-kanade-images/cohn-kanade-images/**/**/*.png'):
        #
        #     dirName = os.path.dirname(filepath)
        #     length = len(glob.glob(dirName+"/*.png"))
        #     diff = abs(length-20)
        #     count = 0
        #     if(length <20 ):
        #          count = 0
        #          for filepath in glob.glob(dirName+"/*.png"):
        #              fileName = os.path.basename(filepath)
        #              fileNameArray = fileName.split(".")
        #              actualFileName = fileNameArray[0]
        #              copyfile(filepath, dirName+"/"+actualFileName+"copy"+str(count)+"."+fileNameArray[1])
        #              count+=1
        #              if count == diff:
        #                  break
        #     if length> 20:
        #         count = 0
        #         for filepath in glob.glob(dirName + "/*.png"):
        #             os.remove(filepath)
        #             count+=1
        #             if count == diff:
        #                 break

        # for filepath in glob.iglob('C:/RIT-Stuff/Topics in System/Project/extended-cohn-kanade-images/cohn-kanade-images/**/**/*.png'):
        #     print(filepath)
        #     fileName = os.path.basename(filepath)
        #     dir = fileName.split("_")
        #     dirPath = os.path.join("outFolder",dir[0],dir[1])
        #     cli(filepath,dirPath)

        print("--Done with mask Image")


        # finalList = np.array(list)
        # print(finalList)
        # count = 0
        # for filepath in glob.glob('C:/Users/asmit/PycharmProjects/pytorch/739-proj/outFolder/**/**/landmark/*.csv'):
        #     dirName = os.path.dirname(filepath)
        #     length = len(glob.glob(dirName + "/*.csv"))
        #
        #     if(length == 20):
        #         count+=1
        #         df = pd.read_csv(filepath)
        #         df = df.drop(df.columns[0], axis=1)
        #         dataFrame = df.to_numpy()
        #         flattenArray = np.ndarray.flatten(dataFrame)
        #         fileName = os.path.basename(filepath)
        #         dir = fileName.split("_")
        #         dirPath = os.path.join("C:/RIT-Stuff/Topics in System/Project/Emotion_labels/Emotion/",dir[0],dir[1])
        #         text = 0
        #         for filepath in glob.glob(dirPath+"/*.txt"):
        #
        #             file1 = open(filepath, "r+")
        #             splitVal = file1.read().split(".")
        #             text = int(splitVal[0])
        #         labelledList = np.append(text,flattenArray)
        #         list.append(labelledList)
        # print(count)
        # finalList = np.array(list)
        # print(finalList)

        countOfLess = 0
        countOfGood = 0
        for filepath in glob.glob('C:/Users/asmit/PycharmProjects/pytorch/739-proj/outFolder/**/**/landmark/'):
            dirName = os.path.dirname(filepath)
            length = len(glob.glob(dirName+"/*.csv"))
            lenV = 0
            for x in  glob.glob(dirName+"/*.csv"):
                lenV+=1
            if(lenV < 20) :
                countOfLess+=1
            else:
               countOfGood+=1

        print(countOfLess)
        print(countOfGood)





