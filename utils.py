import os
from termcolor import colored
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw, ImageFont


class Logger:
    def __init__(self, color):
        self.color = color
        
    def log(self, key, value):
        """
        Prints key with the color of the instance and value with the regular color, on a new line
        """
        print(colored(text=key, color=self.color))
        print(value)

    @classmethod
    def set_color(cls, text, color):
        res = colored(text, color)
        return res

    @classmethod
    def enable_colors(cls):
        os.system('color')
        print(colored("Colors enabled", "green"))

def parse_xml_classification(file):
    tree = ET.parse(file)
    root = tree.getroot()
    res = []
    for obj in root.iter("object"):
        name = obj.find("name").text
        res.append([name])
    return res

def parse_xml_detection(file):
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

def draw_bndbox(base, seq):
    name, xmin, xmax, ymin, ymax = seq
    # Create image
    rec = Image.new("RGBA", base.size, (255, 255, 255, 0))
    # Create drawing context
    d = ImageDraw.Draw(rec)
    # Create rectangle
    d.rectangle((xmin, ymin, xmax, ymax), fill=None, outline="red", width=2)
    
    # Get a font
    font = ImageFont.truetype("arial.ttf", size=15)
    # Create text
    d.text((xmin, ymin), name, font=font, fill=(255, 255, 255, 255))
    # Merge base with the rectangle
    out = Image.alpha_composite(base, rec)
    return out

def annotate_image(image, annotation):
    """
    For a given image path and a given annotation path, annotate the image and returns the modified version
    """
    desc = parse_xml_detection(annotation)
    base = Image.open(image).convert("RGBA")
    for seq in desc:
        base = draw_bndbox(base, seq)
    return base

