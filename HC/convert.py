import numpy as np
from pathlib import Path
import sys
import gzip
import pandas as pd
import matplotlib.pyplot as plt


with gzip.open('C:/Users/bincl/BA-Thesis/Dataset/1_20000_nopos.gz','rt', encoding='utf-8') as input:
    for line in input:
        values = line.strip().split("\t")
        word = values[0]
        data = [entry.split(",") for entry in values[1:]]
        year = {entry[0]: entry[1] for entry in data}
        print(year)
