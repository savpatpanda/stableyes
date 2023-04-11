import customtkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
from skimage import measure
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import random
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import scipy
from sklearn.metrics import r2_score
import time
import pandas as pd
import joblib
from PIL import Image, ImageTk

global image_frame
global selectedFilename
global patientVar
global size 
global retina

size = (0, 0)
selectedFilename = ""
mac = None
retina = None

def size_det(size, factor):
	w_img = size[0]
	h_img = size[1]
	global image_frame
	if w_img > image_frame.winfo_width():
		h_img = int(float(h_img) * (image_frame.winfo_width() - 40) / w_img)
		w_img = image_frame.winfo_width() - 40
	if h_img > image_frame.winfo_height() * factor:
		w_img = int(float(w_img) * (image_frame.winfo_height() * factor - 20) / h_img)
		h_img = int(image_frame.winfo_height() * factor - 20)
	return (w_img, h_img)

def select_file():
  global selectedFilename 
  global patientVar
  global image_frame
  global size
  global image_label

  filetypes = (
    ('text files', '*.jpeg'),
    ('All files', '*.*')
  )

  if selectedFilename == "":
    filename = fd.askopenfilename(
      title='Open a file',
      initialdir='/',
      filetypes = filetypes)
    selectedFilename = filename

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(frameon = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    fig.savefig('unlabeled_img.jpeg')

    #image = ImageTk.PhotoImage(file = 'unlabeled_img.jpeg')
    #imagebox.config(image=image)
    #imagebox.image = image # save a reference of the image to avoid garbage collection

    unlabeled_img = Image.open("unlabeled_img.jpeg")
    size = size_det(unlabeled_img.size, 0.5)
    unlabeled_img_packed = tk.CTkImage(unlabeled_img, size = size)
    image_label = tk.CTkLabel(image_frame, text="", image=unlabeled_img_packed, fg_color = 'blue')
    image_label.grid(row=0, column=0, padx=20, pady=10)

    num = random.randint(10, 90)
    patientVar.configure(text = "Patient 00" + str(num) + " (DoB: 12/16/2000, Record Date: 03/13/2023 04:49:03 PM)")

def analyze():
  global selectedFilename
  global mac
  global x1
  global size
  global image_frame
  global image_label
  global retina

  img = cv2.imread(selectedFilename, cv2.IMREAD_GRAYSCALE)

  fig = plt.figure(frameon = False)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  plt.axis('off')

  rf = joblib.load("rf.pkl")
  img, contours = findContour(selectedFilename)
  features = pd.DataFrame(contourSetExtract(contours))
  features = features.drop(['num'], axis = 1)
  predictions = rf.predict(features)
  for i in range(len(predictions)):
    if predictions[i] > 0.5:
      cont = contours[i]
      x1 = cont[1]
      y1 = cont[0]
      ax.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
      ax.scatter(x1,y1)
      retina = features.iloc[[i]]
  maxAvgVal = -1
  contMax = None
  for cont in contours:
    sum = 0.0
    for i in range(len(cont[0])):
      sum += img.item(int(cont[0][i]), int(cont[1][i]))
    avgVal = float(sum) / len(cont[0])
    if avgVal > maxAvgVal:
       maxAvgVal = avgVal
       contMax = cont
  x2 = contMax[1]
  y2 = contMax[0]
  ax.scatter(x2,y2)
  plt.axis('off')
  fig.savefig('labeled_img.jpeg')
  #plt.close(fig)

  labeled_img = Image.open("labeled_img.jpeg")
  labeled_img_packed = tk.CTkImage(labeled_img, size = size)
  image_label.configure(image=labeled_img_packed)

  mac = macThickness(x1, y1, x2, y2)

def checkFocus():
  global selectedFilename
  img = cv2.imread(selectedFilename, cv2.IMREAD_GRAYSCALE)

  foc = focus(img)
  if foc > 1000:
    showinfo(
      title='Image Focus',
      message="Image Focus: Good (<1000)"
    )
  else:
    showinfo(
      title='Image Focus',
      message="Image Focus: Poor (>1000)"
  )

def macThickness(x1, y1, x2, y2):
  arr = []
  for val in range(len(x1)):
    pos = findClosest(x1[val], x2)
    arr.append((y2[pos] - y1[val])*10)
  return arr

def findClosest(x, xArr):
  minDis = 1E9
  minVal = -1
  for i in range(len(xArr)):
    if (x - xArr[i]) < minDis:
      minDis = x - xArr[i]
      minVal = i
  return minVal

def findContour(img_path):
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  img_blur = cv2.GaussianBlur(img, (11,11), 0)
  img_blur_2 = cv2.GaussianBlur(img, (25, 25), 0)
  sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=2, ksize=5)
  sobely_2 = cv2.Sobel(src=img_blur_2, ddepth=cv2.CV_64F, dx=0, dy=2, ksize=5)
  contours = measure.find_contours(sobely, 0.1)
  contours_2 = measure.find_contours(sobely_2, 0.1)
  linearized = []

  for cont in contours:
    if len(cont) > 300:
      x = [point[0] for point in cont]
      y = [point[1] for point in cont]
      if (max(y) - min(y)) > 0.8*img.shape[1]:
        linearized.append([x,y])

  for cont in contours_2:
    if len(cont) > 300:
      x = [point[0] for point in cont]
      y = [point[1] for point in cont]
      if (max(y) - min(y)) > 0.8*img.shape[1]:
        linearized.append([x,y])

  return img, linearized

def contourFeatureExtraction(contour, num):
  fit = np.polyfit(contour[1] - np.average(contour[1]), contour[0] - np.average(contour[0]), 20)
  poly = np.poly1d(fit)
  r2 = r2_score(contour[0] - np.average(contour[0]), poly(contour[1] - np.average(contour[1])))
  stdX = np.std(contour[1])
  minX = np.min(contour[1])
  maxX = np.max(contour[1])
  rangeX = maxX - minX
  avgX = np.average(contour[1])
  lengthX = len(contour[1])
  stdY = np.std(contour[0])
  minY = np.min(contour[0])
  maxY = np.max(contour[0])
  rangeY = maxY - minY
  avgY = np.average(contour[0])
  lengthY = len(contour[0])
  summaryStats = [r2, stdX, minX, maxX, rangeX, avgX, lengthX, stdY, minY, maxY, rangeY, avgY, lengthY, num]
  return np.concatenate([fit, summaryStats])

def contourSetExtract(contourList):
  collect = []
  for i in range(len(contourList)):
    features = contourFeatureExtraction(contourList[i], i)
    collect.append(features)
  collect = np.transpose(collect)
  keys = ['c20', 'c19', 'c18', 'c17', 'c16', 'c15', 'c14', 'c13', 'c12', 'c11', 'c10', 'c9', 'c8', 'c7', 'c6', 'c5', 'c4', 'c3', 'c2', 'c1', 'c0',
      'r2', 'stdX', 'minX', 'maxX', 'rangeX', 'avgX', 'lengthX', 'stdY', 'minY', 'maxY', 'rangeY', 'avgY', 'lengthY', 'num']
  collect = {k:v for k,v in zip(keys,collect)}
  return collect

def focus(img):
	return cv2.Laplacian(img, cv2.CV_64F).var()

def mac():
  global mac
  global x1
  global image_frame
  fig = plt.figure(figsize = (4.5, 3))
  plt.tight_layout()
  plt.xlabel('Horizontal Image Position (Pixels)')
  plt.ylabel('Depth (um)')
  plt.axis('on')
  plt.scatter(x1, mac, s = 3)
  fig.savefig('macThickness.jpeg', bbox_inches='tight')

  macThicknessGraph = Image.open("macThickness.jpeg")
  mac_img_packed = tk.CTkImage(macThicknessGraph, size = size_det(macThicknessGraph.size, 0.4))
  mac_image_label = tk.CTkLabel(image_frame, text="", image=mac_img_packed)
  mac_image_label.grid(row=1, column=0, padx=20, pady=10)

def diag():
  global retina
  disModel = joblib.load("diseaseModel.pkl")
  diagnosis = disModel.predict(retina)
  diagnosis = diagnosis[0]
  diag_label.configure(text = diagnosis)

tk.set_appearance_mode("light")
tk.set_default_color_theme("blue")

root = tk.CTk()
#root.geometry("1000x600")
root.title("StablEyes OCT Image Analysis")
root.attributes('-fullscreen', True)

# set grid layout 1x2
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

button_frame = tk.CTkFrame(master = root, corner_radius=0, fg_color = "#d8e1e8")
button_frame.grid(row=0, column=0, sticky="nsew")
button_frame.grid_rowconfigure(list(range(7)), weight=1)

image_frame = tk.CTkFrame(master = root, corner_radius=0, fg_color = 'white')
image_frame.grid(row=0, column=1, sticky="nsew")
image_frame.grid_rowconfigure(2, weight=1)
image_frame.grid_columnconfigure(0, weight=1)

logo_img = Image.open("logo_stableyes.png")
logo_img_packed = tk.CTkImage(logo_img, size = (300, 50))
logo_label = tk.CTkLabel(button_frame, text="", image=logo_img_packed, fg_color = 'transparent')
logo_label.grid(row=0, column=0, padx=20, pady=10, sticky = "nsew")

file_button = tk.CTkButton(master = button_frame, 
							corner_radius = 8, 
							border_width = 0,
							text = "  Open File  ", 
              font=tk.CTkFont("Roboto", 24), 
              fg_color = "#98bad5",
              border_color = "#b2cbde",
              text_color = "#304674",
							command = select_file)
file_button.grid(row = 1, column = 0, padx = 15, pady = 35, sticky = "nsew")

focus_button = tk.CTkButton(master = button_frame, 
              corner_radius = 8, 
              border_width = 0,
              text = "  Check Focus  ", 
              font=tk.CTkFont("Roboto", 24),
              fg_color = "#98bad5", 
              border_color = "#b2cbde",
              text_color = "#304674",
              command = checkFocus)
focus_button.grid(row = 2, column = 0, padx = 15, pady = 35, sticky = "nsew")


label_button = tk.CTkButton(master = button_frame, 
							corner_radius = 8, 
							border_width = 0,
							text = "  Label Image  ",  
              font=tk.CTkFont("Roboto", 24),
              fg_color = "#98bad5",
              border_color = "#b2cbde",
              text_color = "#304674",
							command = analyze)
label_button.grid(row = 3, column = 0, padx = 15, pady = 35, sticky = "nsew")

mac_button = tk.CTkButton(master = button_frame, 
							corner_radius = 8, 
							border_width = 0,
							text = "  Macular Analysis  ", 
              font=tk.CTkFont("Roboto", 24),
              fg_color = "#98bad5",
              border_color = "#b2cbde",
              text_color = "#304674",
							command = mac)
mac_button.grid(row = 4, column = 0, padx = 15, pady = 35, sticky = "nsew")

diag_button = tk.CTkButton(master = button_frame, 
              corner_radius = 8, 
              border_width = 0,
              text = "  Diagnose  ", 
              font=tk.CTkFont("Roboto", 24),
              fg_color = "#98bad5",
              border_color = "#b2cbde",
              text_color = "#304674",
              command = diag)
diag_button.grid(row = 5, column = 0, padx = 15, pady = 35, sticky = "nsew")

diag_label = tk.CTkLabel(master = button_frame,
                          text = "",
                          font=tk.CTkFont("Roboto", 24, "bold"),
                          width = 300)
diag_label.grid(row = 6, column = 0, padx = 15, pady = 35, sticky = "nsew")


patientVar = tk.CTkLabel(master = image_frame,
							corner_radius = 8,
							text = "")
patientVar.grid(row = 2, column = 0, padx = 10, pady = 20, sticky = "sew")

root.mainloop()