"""
shape.py - A multi-triangle shape picked from Google color palette
"""

import random
from typing import List, Tuple
from config import EnvConfig, VisConfig

GOOGLE_COLORS = VisConfig().GOOGLE_COLORS

class Shape:
    def __init__(self)->None:
        self.triangles:List[Tuple[int,int,bool]]=[]
        self.color:Tuple[int,int,int]=random.choice(GOOGLE_COLORS)
        self._generate()

    def _generate(self)->None:
        n = random.randint(1,5)
        first_up = random.choice([True,False])
        self.triangles.append((0,0,first_up))
        for _ in range(n-1):
            lr,lc,lu=self.triangles[-1]
            nbrs=self._neighbors(lr,lc,lu)
            if nbrs:self.triangles.append(random.choice(nbrs))

    def _neighbors(self,r:int,c:int,up:bool):
        if up:
            ns=[(r,c-1,False),(r,c+1,False),(r+1,c,False)]
        else:
            ns=[(r,c-1,True),(r,c+1,True),(r-1,c,True)]
        return [x for x in ns if x not in self.triangles]

    def bbox(self)->Tuple[int,int,int,int]:
        rr=[t[0] for t in self.triangles]
        cc=[t[1] for t in self.triangles]
        return (min(rr),min(cc),max(rr),max(cc))
