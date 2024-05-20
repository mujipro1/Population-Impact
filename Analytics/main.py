import joblib
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_graph():
    x_feature = x_dropdown.get()
    y_feature = y_dropdown.get()
    type_f = type_dropdown.get()
    x_values = data[x_feature]
    y_values = data[y_feature]
    ax.clear()
    if type_f == 'Scatter':
        ax.scatter(x_values, y_values)
    else:
        ax.plot(x_values, y_values)
    ax.set_xlabel(x_feature)  # Apply custom font
    ax.set_ylabel(y_feature)  # Apply custom font
    
    ax.set_title(f"{y_feature} vs {x_feature}")  # Apply custom font
    canvas.draw()

# Example data (replace with your own data)
data = pd.read_excel("Refined/CleanData.xlsx")



window = Tk()
window.title("Graph Explorer")

# Define custom font style
myFont = tkFont.Font(family="Arial", size=15)
window.option_add("*Font", myFont)

width = window.winfo_screenwidth()
height = window.winfo_screenheight()

leftframe = Frame(window, width=width/2, height=height)
leftframe.pack(side=LEFT)
leftframe.pack_propagate(0)


frame = Frame(leftframe)
frame.pack(side=TOP)
x_label = Label(frame, text="Select X axis feature:", font=myFont)
x_label.pack(side=LEFT)
x_dropdown = ttk.Combobox(frame, values=list(data.keys()), font=myFont)
x_dropdown.pack(side=LEFT)

# Dropdown for Y axis
frame = Frame(leftframe)
frame.pack(side=TOP)
y_label = Label(frame, text="Select Y axis feature:", font=myFont)
y_label.pack(side=LEFT)
y_dropdown = ttk.Combobox(frame, values=list(data.keys()), font=myFont)
y_dropdown.pack(side=LEFT)

frame = Frame(leftframe)
frame.pack(side=TOP)
type_label = Label(frame, text="Select Type of Graph:", font=myFont)
type_label.pack(side=LEFT)
type_dropdown = ttk.Combobox(frame, values=['Scatter', 'Line'], font=myFont)
type_dropdown.pack(side=LEFT)

graph_button = Button(leftframe, text="Create Graph", command=create_graph, font=myFont)
graph_button.pack(side=TOP)

fig, ax = plt.subplots(figsize=(9, 8))
canvas = FigureCanvasTkAgg(fig, master=leftframe)
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

rightframe = Frame(window, width=width/2, height=height)
rightframe.pack(side=RIGHT)
rightframe.pack_propagate(0)

def get_data():
    year = int(entry_year.get())
    unemployment, population, gdp = get_data_for_year(year)
    unemployment_label.config(text=f"Unemployment: {unemployment}")
    population_label.config(text=f"Population:   {population}")
    gdp_label.config(text=f"GDP:          {gdp}")

def get_data_for_year(year):
    
    # model = joblib.load('Models/gdp.pkl')
    # gdp = np.exp(model.predict(([[year]])))

    # model = joblib.load('Models/unemployment.pkl')
    # unemployment = np.exp(model.predict(([[year]])))

    model = joblib.load('Models/population.pkl')
    population = model.predict([[year]])
    
    return "unemployment", population, "gdp"


frame = Frame(rightframe)
frame.pack(side=TOP)
year_label = Label(frame, text="Enter Year:", font=myFont)
year_label.pack(side=LEFT)
entry_year = Entry(frame, font=myFont)
entry_year.pack(side=LEFT)

predict_button = Button(rightframe, text="Predict", command=get_data, font=myFont)
predict_button.pack(side=TOP)

frame = Frame(rightframe, width=300, height=30)
frame.pack(side=TOP)
frame.pack_propagate(0)
gdp_label = Label(frame, text="GDP", font=myFont)
gdp_label.pack(side=LEFT)

frame = Frame(rightframe, width=300, height=30)
frame.pack(side=TOP)
frame.pack_propagate(0)
unemployment_label = Label(frame, text="Unemployment", font=myFont)
unemployment_label.pack(side=LEFT)

frame = Frame(rightframe, width=300, height=30)
frame.pack(side=TOP)
frame.pack_propagate(0)
population_label = Label(frame, text="Population", font=myFont)
population_label.pack(side=LEFT)

window.mainloop()
