import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('imagePath')
parser.add_argument('labelPath')
args = parser.parse_args()

with open('annotationCSV.csv', 'a') as csvF:
    for i in os.walk(args.imagePath):
        for imageName in i[2]:
            if imageName == '.DS_Store': continue
            imageId = imageName.split('.')[0]
            labelName = imageId + '.txt'
            labelPath = os.path.join(args.labelPath, labelName)
            imagePath = os.path.join(args.imagePath, imageName)
            with open(labelPath, 'r') as labelF:
                for box in labelF.readlines():
                    box = box.split(',')
                    if len(box) != 4:
                        continue
                    # x1, y1, x2, y2
                    csvRecord = [imagePath, box[0], box[1], box[2], str(int(box[3])), 'scratch']
                    csvRecord = ",".join(csvRecord) + '\n'
                    csvF.write(csvRecord)
                



