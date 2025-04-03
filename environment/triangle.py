"""
triangle.py - Single cell with occupant color
"""

from typing import Tuple, Optional, List

class Triangle:
    def __init__(self, row: int, col: int, is_up: bool, is_death: bool=False):
        self.row = row
        self.col = col
        self.is_up = is_up
        self.is_death = is_death
        self.is_occupied = is_death
        self.color: Optional[Tuple[int,int,int]] = None

    def get_points(self, ox:int, oy:int, cw:int, ch:int)->List[Tuple[float,float]]:
        x=ox+self.col*(cw*0.75)
        y=oy+self.row*ch
        if self.is_up:
            return [(x,y+ch),(x+cw,y+ch),(x+cw/2,y)]
        else:
            return [(x,y),(x+cw,y),(x+cw/2,y+ch)]