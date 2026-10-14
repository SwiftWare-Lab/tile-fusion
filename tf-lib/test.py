from typing import TypedDict, Unpack

class ParallelConfig:
    i_b: str
    i_t: str
    cache: bool

def parallel(**kwargs:Unpack[ParallelConfig]):
    print(kwargs["i_b"])

parallel(i_b="i")