# %%
import os
import sys
import pathlib




%reload_ext autoreload
%autoreload 2

# %%

from scripts.preprocess import read_text

# %%

lang = 'en'

path_to_text = r'D:\Uni\Master\FU Data Science\Lectures\SS_21\NLP\project\task_1\data\Harry_en.txt'

text = read_text(path_to_text)
text = text.replace("\n","")



# %%

# %%