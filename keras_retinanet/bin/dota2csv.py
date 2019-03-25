import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('imagePath')
parser.add_argument('labelPath')
args = parser.parse_args()
classes = []

with open('annotationCSV.csv', 'a') as csvF:
    for i in os.walk(args.imagePath):
        for imageName in i[2]:
            imageId = imageName.split('.')[0]
            labelName = imageId + '.txt'
            labelPath = os.path.join(args.labelPath, labelName)
            imagePath = os.path.join(args.imagePath, imageName)
            with open(labelPath, 'r') as labelF:
                for box in labelF.readlines():
                    box = box.split(' ')
                    if len(box) != 10:
                        continue
                    # x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
                    x1 = str(int(min(float(box[0]), float(box[2]), float(box[4]), float(box[6]))))
                    x2 = str(int(max(float(box[0]), float(box[2]), float(box[4]), float(box[6]))))
                    y1 = str(int(min(float(box[1]), float(box[3]), float(box[5]), float(box[7]))))
                    y2 = str(int(max(float(box[1]), float(box[3]), float(box[5]), float(box[7]))))
                    csvRecord = [imagePath, x1, y1, x2, y2, box[8]]
                    csvRecord = ",".join(csvRecord) + '\n'
                    csvF.write(csvRecord)
                    if box[8] not in classes:
                        classes.append(box[8])

with open('classCSV.csv', 'w') as csvF:
    for k, v in enumerate(classes):
        csvF.write(str(v) + ',' + str(k) + '\n')