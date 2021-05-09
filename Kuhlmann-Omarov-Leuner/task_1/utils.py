from typing import List
from tabulate import tabulate

strs = List[str]
ints = List[int]


def read_file(file: str) -> strs:
    with open(file, encoding='utf-8') as reader:
        return [l.strip() for l in reader.readlines()]


def delete_substrings(str_: str, substrings: strs) -> str:
    for substr in substrings:
        str_ = str_.replace(substr, '')
    return str_


def filter_indices(l: List, indices: ints) -> List:
    return [l[i] for i in indices]


def flatten(l: List) -> List:
    return [item for sublist in l for item in sublist]


def print_table(table: List[List]):
    print(tabulate(table))