# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:33:36 2022

@author: tsiragy
"""

import pickle

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v



from tkinter import *
from tkinter import filedialog

def get_file_path():
    global file_path
    # Open and return file path
    file_path= filedialog.askopenfilename(title = "Select A File")
    l1 = Label(window, text = "File path: " + file_path).pack()

window = Tk()
# Creating a button to search the file
b = Button(window, text = "Open File", command = get_file_path).pack()
window.mainloop()
print(file_path)

filename = file_path #get the .mat file with the biomech data

import pandas as pd
filepaths = pd.DataFrame()
data = load(filename)