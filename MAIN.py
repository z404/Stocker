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
    def predict():
        pass
    def back():
        clear_window(root)
        mainmenu()
    Title_Label = tk.Label(root, text = 'Predictor', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Frame = tk.Frame(root, pady=20)
    Frame.pack()
    Company_Label = tk.Label(Frame, text='Company Name:', font = 'Ariel 25 bold')
    Company_Label.grid(row=0,column=0)
    Company_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Company_Entry.grid(row=0, column=1, padx=10, pady = 15)
    Days_Label = tk.Label(Frame, text='Number of days to predict:', font = 'Ariel 25 bold')
    Days_Label.grid(row=1,column=0, padx = 10, pady = 15)
    Days_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Days_Entry.grid(row=1,column=1)
    Predict_Button = tk.Button(root, text = 'Predict!', font = 'Ariel 25 bold', command = predict, pady = 10, width = 25)
    Predict_Button.pack()
    Back_Button = tk.Button(root, text = 'Back', font = 'Ariel 25 bold', command = back, pady = 10, width = 25)
    Back_Button.pack()

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
    Title_Label = tk.Label(root, text = 'Stocker', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Catchphrase_Label = tk.Label(root, text = 'Stalking your Stocks!', font='Ariel 15', pady = 20)
    Catchphrase_Label.pack()
    Start_Button = tk.Button(root, text = 'Start', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = start_button)
    Start_Button.pack()
    Update_Button = tk.Button(root, text = 'Update',font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = update_button)
    Update_Button.pack()
    Exit_Button = tk.Button(root, text = 'Exit', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = exit_button)
    Exit_Button.pack()
mainmenu()
