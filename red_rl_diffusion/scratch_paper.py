import numpy as np
import copy
class Animal(object):
    def __init__(self, cat) -> None:
        self.cat = cat

    def print_cat_whiskers_num(self):
        print("This cat has %d number of whiskers."%self.cat.get_whisker_num())

class cat(object):
    def __init__(self, whisker_num) -> None:
        self.whisker_num = whisker_num

    def get_whisker_num(self):
        return self.whisker_num

    def set_whisker_num(self, num):
        self.whisker_num = num
        return

if __name__ == "__main__":
    cat1 = cat(whisker_num=3)
    animal = Animal(cat1)
    animal.print_cat_whiskers_num()
    cat2 = cat(whisker_num=5)
    cat1 = cat2
    # cat1 = copy.deepcopy(cat2)
    # cat1.whisker_num = 99
    # cat1.set_whisker_num(100)
    animal.print_cat_whiskers_num()
