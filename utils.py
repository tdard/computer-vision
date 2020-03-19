import os
from termcolor import colored


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