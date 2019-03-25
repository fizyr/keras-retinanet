import argparse
import os,shutil
import xml.dom.minidom
import json

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('annotationPath')
#parser.add_argument('labelPath')
args = parser.parse_args()

imageCount = 0

with open(args.annotationPath, 'r') as annotationF:
    annotations = annotationF.readlines()
    for i in annotations:
        i = i.split("'")
        if len(i) < 3: continue
        imageOriginPath = i[0].split(',')[0]
        imageNewPath = 'P'+str(imageCount)+'.jpg'
        imageNewPath = os.path.join(os.path.split(imageOriginPath)[0], '..', 'images', imageNewPath)
        shutil.copyfile(imageOriginPath, imageNewPath)
        labelNewPath = 'P'+str(imageCount)+'.txt'
        labelNewPath = os.path.join(os.path.split(imageOriginPath)[0], '..', 'labels', labelNewPath)
        objects = json.loads(i[1])
        imageCount += 1
        with open(labelNewPath, 'w') as labelF:
            for o in objects:
                print(o)
                x1 = str(int(o['coordinates']['x'] - o['coordinates']['width'] / 2))
                x2 = str(int(o['coordinates']['x'] + o['coordinates']['width'] / 2))
                y1 = str(int(o['coordinates']['y'] - o['coordinates']['height'] / 2))
                y2 = str(int(o['coordinates']['y'] + o['coordinates']['height'] / 2))
                box = [x1, y1, x2, y2, '']
                labelF.write(','.join(box)+'\n')


        print(objects)