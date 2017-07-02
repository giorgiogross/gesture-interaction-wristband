# -*- coding: utf-8 -*-
"""
Created on Wed May 24 22:14:48 2017

@author: Richard
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:16:55 2017

@author: Richard
Read pickle files with the equity price data and save them in a xlsx file
"""
from Tkinter import *
import ttk
import matplotlib
from matplotlib.figure import Figure
import sys
import os
import inspect
import numpy as np
# Working with excel sheets
import pandas as pd
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from PIL import ImageTk, Image
from scanner.Gestures import Gestures


class Dashboard:
    cb1 = 0

    def init(self):
        wd = os.path.dirname(os.path.abspath(inspect.stack()[0][1])) 

        x_es = pd.read_excel(wd+'/es_data.xlsx')

        x_es=x_es[np.isfinite(x_es['price'])]
        x_es=x_es[['price']]

        x_bank = pd.read_excel(wd+'/bank_data.xlsx')
        x_bank=x_bank[np.isfinite(x_bank['price'])]
        x_bank=x_bank[['price']]
        x=x_es.index.values
        z=x_es['price']
        y=x_bank['price']

        #Canvas

        root = Tk()
        root.wm_title("Dashboard")

        board = PanedWindow(master=root,orient=VERTICAL)
        board.pack()

        m1=PanedWindow(board)
        m2=PanedWindow(board,orient=HORIZONTAL)


        #Image
        im=Image.open(wd+"/kpmg.jpg")
        im = im.resize((150,100), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(im)
        panel = Label(m1, image = img)
        panel.pack(side = TOP, fill = "both", expand = "no")

        def handler(eventObject):
            a.clear()
            if (self.cb1.get()=='ES Banks'):
                a.plot(x,y,'g')
            elif (self.cb1.get()=='ES 50'):
                a.plot(x,z,'g')
            else:
                a.plot(x,z,'g',x,y,'b')
            f.canvas.draw_idle()
        #Combobox
        values=['All','ES 50','ES Banks']
        labels=dict((value,Label(m1, text=value))for value in values)
        #Combobox
        cbp1 = ttk.Labelframe(m1, text='Set Graphs')
        self.cb1 = ttk.Combobox(cbp1,values=values,state='readonly')
        self.cb1.current(0)
        self.cb1.bind('<<ComboboxSelected>>', handler)
        self.cb1.pack(pady=5, padx=10)
        cbp1.pack(in_=m1, side=TOP, pady=5, padx=10)


        #Display Figure
        f = Figure(figsize=(9, 3), dpi=100)
        a = f.add_subplot(111)

        a.plot(x,z,'g',x,y,'b')

        canvas = FigureCanvasTkAgg(f, master=m1)
        canvas.show()
        canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)

        #toolbar = NavigationToolbar2TkAgg(canvas, root)
        #toolbar.update()
        canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=1)



        '''
        def on_key_event(event):
            print('you pressed %s' % event.key)
            key_press_handler(event, canvas, toolbar)

        canvas.mpl_connect('key_press_event', on_key_event)
        '''

        def _quit():
            root.quit()     # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent
                            # Fatal Python Error: PyEval_RestoreThread: NULL tstate
            sys.exit()

        def radio():
            ax.clear()
            bx.clear()
            if(var.get()=="2010"):
                ax.pie(data_2010,labels=labels,autopct='%1.1f%%',shadow=False, startangle=90)
                bx.bar(y_pos, data_2010, align='center', alpha=0.8, color ='r')
                bx.set_xticklabels(['','ESX Banks', 'ESX Oil', 'ESX Automobil','ESX Ins', 'ESX FS'])
            elif(var.get()=="2011"):
                ax.pie(data_2011,labels=labels,autopct='%1.1f%%',shadow=False, startangle=90)
                bx.bar(y_pos, data_2011, align='center', alpha=0.8, color ='r')
                bx.set_xticklabels(['','ESX Banks', 'ESX Oil', 'ESX Automobil','ESX Ins', 'ESX FS'])
            else:
                ax.pie(data_2012,labels=labels,autopct='%1.1f%%',shadow=False, startangle=90)
                bx.bar(y_pos, data_2012, align='center', alpha=0.8, color ='r')
                bx.set_xticklabels(['','ESX Banks', 'ESX Oil', 'ESX Automobil','ESX Ins', 'ESX FS'])
            pie.canvas.draw_idle()
            bar.canvas.draw_idle()

        #Bottom Pane
        m3=PanedWindow(m2,orient=HORIZONTAL)
        m4=PanedWindow(m2,orient=VERTICAL)
        #Left Checkboxes
        modes=[("2010", "2010"),
                ("2011", "2011"),
                ("2012", "2012")]
        var = StringVar()
        var.set('L')

        b1=Radiobutton(m4,text='2010',variable=var,value='2010',command = radio)
        b1.pack(side=TOP,anchor=W)
        b1.select()
        b2=Radiobutton(m4,text='2011',variable=var,value='2011',command = radio)
        b2.pack(side=TOP,anchor=W)
        b3=Radiobutton(m4,text='2012',variable=var,value='2012',command = radio)
        b3.pack(side=TOP,anchor=W)
        '''
        for text, mode in modes:
            b[]=Radiobutton(m3,text=text,variable=var,value=mode,command = radio)
            b[].pack(anchor=W)
        '''
        #c = Checkbutton(m2, text="Color image", variable=var,onvalue="RGB", offvalue="L")
        #c.pack(side=LEFT)
        #Left Pie Chart
        labels=['Banks','Oil','Automobil','Insurance','Financial Services']
        data_2010=[400,347,215,520,174]
        data_2011=[310,360,225,397,239]
        data_2012=[215,470,443,315,453]

        pie = Figure(figsize=(5, 7), dpi=100)
        ax = pie.add_subplot(111)
        ax.pie(data_2010,labels=labels,autopct='%1.1f%%',shadow=False, startangle=90)
        canvas2 = FigureCanvasTkAgg(pie, master=m3)
        canvas2.show()
        canvas2.get_tk_widget().pack(side=RIGHT, fill=NONE, expand=0)


        #Checkbuttons for Barchart
        '''
        modes2=[("ESX Banks", "ESX Banks"),
                ("ESX Oil", "ESX Oil"),
                ("ESX Automobil", "ESX Automobil"),
                ("ESX Ins", "ESX Ins"),
                ("ESX FS", "ESX FS")]
        for text, mode in modes2:
            c=Checkbutton(m4,text=text)
            c.pack(anchor=W)
        '''
        #BarChart Left
        objects = ('ESX Banks', 'ESX Oil', 'ESX Automobil','ESX Ins', 'ESX FS')
        y_pos = np.arange(len(objects))
        #
        #plt.figure(figsize=(15,5)) 
        bar = Figure(figsize=(9,2), dpi=100)
        bx = bar.add_subplot(111)
        bx.bar(y_pos, data_2010, align='center', alpha=0.8, color ='r')
        bx.set_xticklabels(['','ESX Banks', 'ESX Oil', 'ESX Automobil','ESX Ins', 'ESX FS'])
        canvas3 = FigureCanvasTkAgg(bar, master=m4)
        canvas3.show()
        canvas3.get_tk_widget().pack(side=TOP, fill=NONE, expand=0)

        #button = Button(master=m2, text='Quit', command=_quit)
        #button.pack(side=RIGHT)
        m2.add(m3)
        m2.add(m4)
        board.add(m1)
        board.add(m2)

        root.mainloop()

    def action(self, action_id):
        if action_id == Gestures.SWIPE_LEFT:
            current = self.cb1.current()
            current = current - 1 if current > 0 else 0
            self.cb1.current(current)
            self.cb1.event_generate("<<ComboboxSelected>>")
        elif action_id == Gestures.SWIPE_RIGHT:
            current = self.cb1.current()
            current = current + 1 if current < 2 else 2
            self.cb1.current(current)
            self.cb1.event_generate("<<ComboboxSelected>>")
