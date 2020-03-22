import os
from termcolor import colored
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from random import choice
import numpy as np
from progressbar import ProgressBar


class Logger(object):
    colors = ["grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    def __init__(self, color):
        self.color = color
        
    def log(self, key, value):
        """
        Prints key with the color of the instance and value with the regular color, on a new line
        """
        print(colored(text=key, color=self.color))
        print(value)

    @classmethod
    def set_color(cls, text, color="blue", random=False):
        if random:
            color = choice(Logger.colors)
        res = colored(text, color)
        return res

    @classmethod
    def enable_colors(cls):
        os.system('color')
        print(colored("Colors enabled", "green"))

class AnnotationParser(object):
    
    tasks = ("classification", "detection", "segmentation")

    @classmethod
    def parse(cls, file, task):
        """
        File is a correct XML path to be opened and task a string corresponding to either classification, detection or segmentation.
        """
        if task not in cls.tasks:
            raise ValueError("The task you provided: '{}' is unknown".format(task))
        
        if task == 'classification':
            res = cls.__parse_for_classification(file)
        elif task == 'detection':
            res = cls.__parse_for_detection(file)
        else:
            res = cls.__parse_for_segmentation(file)
        return res

    @classmethod
    def __parse_for_classification(cls, file):
        tree = ET.parse(file)
        root = tree.getroot()
        res = []
        for obj in root.iter("object"):
            name = obj.find("name").text
            res.append(name)
        return res

    @classmethod
    def __parse_for_detection(cls, file):
        tree = ET.parse(file)
        root = tree.getroot()
        res = []
        for obj in root.iter("object"):
            name = obj.find("name").text
            
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            res.append([name, xmin, xmax, ymin, ymax])
        return res

    @classmethod
    def __parse_for_segmentation(cls, file):
        return []


class PascalVOCExtractor(object):
    @classmethod
    def extract_names(cls, image_names):
        # Retrieve targetted image names
        with open(image_names, "r") as file:
            names = file.readlines()
            names = list(map(lambda x: x.strip(), names))
        return names

    @classmethod
    def extract_paths(cls, images_path, annotations_path, names): 
        # Create full path to images
        images = list(map(lambda x: os.path.join(images_path, "{}.jpg".format(x)), names))

        # Create full path to annotations
        annotations = list(map(lambda x: os.path.join(annotations_path, "{}.xml".format(x)), names))
        return images, annotations


class IOProcessor(object):
    
    tasks = ("classification", "detection", "segmentation")

    @classmethod
    def process(cls, im_paths, ann_paths, input_size, task, classes):
        if task not in cls.tasks:
            raise ValueError("The task you provided: '{}' is unknown".format(task))
        
        if task == 'classification':
            x, y = cls.__process_for_classification(im_paths, ann_paths, input_size, classes)
        elif task == 'detection':
            x, y = cls.__process_for_detection(im_paths, ann_paths, input_size, classes)
        else:
            x, y = cls.__process_for_segmentation(im_paths, ann_paths, input_size, classes)
        return x, y
    
    @classmethod
    def __process_for_classification(cls, im_paths, ann_paths, input_size, classes):
        n = len(im_paths)
        m = len(classes)
        pbar = ProgressBar()
        # Create image and annotations array 
        im_array = np.zeros((n, input_size, input_size, 3), dtype="float16") 
        desc_array = np.zeros((n, m), dtype="int8")
        for k in pbar(range(n)):
            # Image processing
            im = Image.open(im_paths[k])
            w, h = im.size
            # Reshape if necessary
            if max(im.size) > input_size:
                ratio = max(im.size)/input_size
                w = int(w / ratio)
                h = int(h / ratio)
                im = im.resize(size=(w, h), resample=Image.BICUBIC)
            assert max(im.size) <= input_size
            # Cast PIL image in numpy array and normalize features between [0,1]
            im = np.asarray(im)/255 # converts also w,h,c -> h,w,c
            # Bottom and right zero-padding
            im_array[k, :h, :w, :] = im
            
            # Annotations processing
            desc = AnnotationParser().parse(ann_paths[k], "classification")
            col = np.zeros((1, m))
            for c in desc:
                col[:, classes[c]-1] = 1
            desc_array[k, :] = col
        return im_array, desc_array

    @classmethod
    def __process_for_detection(cls, im_paths, ann_paths, input_size, classes):
        return None, None

    @classmethod
    def __process_for_segmentation(cls, im_paths, ann_paths, input_size, classes):
        return None, None


def draw_bndbox(base, seq):
    name, xmin, xmax, ymin, ymax = seq
    # Create image
    rec = Image.new("RGBA", base.size, (255, 255, 255, 0))
    # Create drawing context
    d = ImageDraw.Draw(rec)
    # Create rectangle
    d.rectangle((xmin, ymin, xmax, ymax), fill=None, outline="red", width=2)
    
    # Get a font
    font = ImageFont.truetype("arial.ttf", size=20)
    # Create text
    d.text((xmin, ymin), name, font=font, fill=(255, 255, 255, 255))
    # Merge base with the rectangle
    out = Image.alpha_composite(base, rec)
    return out

def annotate_image(image, annotation):
    """
    For a given image path and a given annotation path, annotate the image and returns the modified version
    """
    desc = AnnotationParser().parse(file=annotation, task="detection")
    base = Image.open(image).convert("RGBA")
    for seq in desc:
        base = draw_bndbox(base, seq)
    return base, desc

def load_data(inputs_path, im, desc):
    images = np.load(os.path.join(inputs_path, im))
    descriptions = np.load(os.path.join(inputs_path, desc))
    return (images, descriptions)
