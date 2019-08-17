#Main project

#importing required packages
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime,math
import tkinter as tk

root = tk.Tk()

def clear_window(window):
    for ele in window.winfo_children():
        ele.destroy()

def start_menu():
    global root
    clear_window(root)

def mainmenu():
    #function to start and initialize the program
    global root
    root.attributes("-fullscreen",True)
    def start_button():
        start_menu()
    def update_button():
        pass
    def exit_button():
        exit(0)
    Title_Label = tk.Label(root, text = 'Stock Price Prediction', font = 'Ariel 40 bold', pady = 50)
    Title_Label.pack()
    Start_Button = tk.Button(root, text = 'Start', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = start_button)
    Start_Button.pack()
    Update_Button = tk.Button(root, text = 'Update',font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = update_button)
    Update_Button.pack()
    Exit_Button = tk.Button(root, text = 'Exit', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = exit_button)
    Exit_Button.pack()
mainmenu()
