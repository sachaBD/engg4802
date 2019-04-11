import time, sys
from IPython.display import clear_output

class ProgressBar():
    
    def __init__(self):
        self.bar_length = 20
        self.current = 0
        
    def set_length(self, length):
        self.length = length
        self.current = 0
        
    def progress(self, step=1):
        self.current += step
        
        progress = self.current / self.length
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1

        block = int(round(self.bar_length * progress))

        clear_output(wait = True)
        text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (self.bar_length - block), progress * 100)
        print(text)