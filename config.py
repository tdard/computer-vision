import os
# The place to be for hardcoded variables!


class Defaults(object):
    def __init__(self):
        self.CLASSES = {
            "aeroplane" : 1,
            "bicycle" : 2,
            "bird" : 3,
            "boat" : 4,
            "bottle" : 5,
            "bus" : 6,
            "car" : 7,
            "cat" : 8,
            "chair" : 9, 
            "cow" : 10,
            "diningtable" : 11,
            "dog" : 12,
            "horse" : 13,
            "motorbike" : 14,
            "person" : 15,
            "pottedplant" : 16,
            "sheep" : 17,
            "sofa" : 18,
            "train" : 19,
            "tvmonitor" : 20
        }      
        cwd = os.getcwd()
        self.ANNOTATIONS_PATH = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations")
        self.IMAGES_PATH = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages")
        self.IMAGES_NAMES = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main\trainval.txt")
        self.OUTPUTS_PATH = os.path.join(cwd, r"data\outputs")
        self.LOGS_PATH = os.path.join(cwd, r"data\logs")
        self.TASK = "classification"
        self.INPUT_SIZE = 224
