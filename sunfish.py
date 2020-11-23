#!/usr/bin/python3

import tkinter as tk
import tkinter.ttk as ttk
import os
import math


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.options = {}
        self.createWidgets()

    def addOptioni(self, key, labelText, x1, x2, defaultValue):
        label = ttk.Label(self.frameOptions, text=labelText)
        spin = ttk.Spinbox(self.frameOptions)
        spin["from"] = x1
        spin["to"] = x2
        spin.set(defaultValue)
        spin.grid(row=self.optionRow, column=2)
        label.grid(row=self.optionRow, column=1)
        self.optionRow = self.optionRow + 1

        self.options[key] = spin

    def addOptionf(self, key, labelText, x1, x2, defaultValue):
        label = ttk.Label(self.frameOptions, text=labelText)
        spin = ttk.Spinbox(self.frameOptions)
        spin["from"] = x1
        spin["to"] = x2
        spin["format"] = "%3.3f"
        spin.set(defaultValue)
        spin.grid(row=self.optionRow, column=2)
        label.grid(row=self.optionRow, column=1)
        self.optionRow = self.optionRow + 1

        self.options[key] = spin

    def createWidgets(self):
        self.btnRender = tk.Button(self)
        self.btnRender["text"] = "Render"
        self.btnRender["command"] = self.render
        self.btnRender.pack(side="top")

        self.frameOptions = ttk.LabelFrame(self)
        self.frameOptions["text"] = "Options"
        self.frameOptions.pack(side="top")
        self.optionRow = 1

        # ray tracer options
        self.addOptioni("-width",  "Width", 320, 2560, 1280)
        self.addOptioni("-height", "Height", 200, 1440, int(1280/2.4))
        self.addOptioni("-rays", "Rays/pixel", 1, 1000, 10)
        self.addOptioni("-samples", "Samples/pixel", 1, 1000, 100)
        self.addOptioni("-depth", "Ray depth", 1, 16, 16)
        self.addOptionf("-exposure", "Exposure", -12, 12, -2)
        self.addOptionf("-gamma", "Gamma", 0.1, 3.0, 1.0)

        self.btnQuit = tk.Button(self)
        self.btnQuit["text"] = "Quit"
        self.btnQuit["command"] = self.master.destroy
        self.btnQuit.pack(side="bottom")

    def render(self):
        print("calling sunfish EXE")
        options = ""
        for k in self.options.keys():
            value = self.options[k].get()
            options = options + " " + k + " " + value

        if os.name == "nt":
            command = ".\\x64\\Release\\sunfish.exe"
        else:
            command = "./x64/Release/sunfish.exe"
        commandLine = command + options

        os.system(commandLine)
        print("all done")


root = tk.Tk()
app = Application(master=root)
app.mainloop()
