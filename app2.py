import numpy as np
import cv2 as cv
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter
from tkinter import filedialog

filepath = "C:\\Users\\habib\\Desktop\\UNI"


def openfile():
    global filepath
    filepath = filedialog.askopenfilename()
    img1 = cv.imread(filepath, 0)
    img1 = cv.resize(img1, (400, 200))
    im1 = Image.fromarray(img1)
    imgtk1 = ImageTk.PhotoImage(image=im1)

    before.configure(image=imgtk1)
    before.image = imgtk1

    after.configure(image=imgtk1)
    after.image = imgtk1


def identity():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))

    im1 = Image.fromarray(img2)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def nagative():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    h, w = img2.shape

    for row in range(h):
        for col in range(w):
            img2[row][col] = 255 - img2[row][col]

    im1 = Image.fromarray(img2)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def logfun():
    num = int(logent.get())
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    h, w = img2.shape

    for row in range(h):
        for col in range(w):
            img2[row][col] = int(num * np.log2(img2[row][col] + 1))

    im1 = Image.fromarray(img2)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def powe():
    num = float(powent.get())
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    h, w = img2.shape

    for row in range(h):
        for col in range(w):
            img2[row][col] = 255 * (img2[row][col] / 255) ** num

    im1 = Image.fromarray(img2)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def contr():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    h, w = img2.shape
    a = int(stechminent.get())
    b = int(strechmaxent.get())
    R = b-a

    for row in range(h):
        for col in range(w):
            img2[row][col] = ((img2[row][col]-a)/R) * 255
            img2[row][col] = np.rint(img2[row][col])

    im1 = Image.fromarray(img2)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def intslic():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    h, w = img2.shape
    img1 = np.zeros((h, w), dtype='uint8')

    min_range = int(minent.get())
    max_range = int(maxent.get())

    for i in range(h):
        for j in range(w):
            if img2[i, j] > min_range and img2[i, j] < max_range:
                img1[i, j] = 255
            else:
                img1[i, j] = 0

    im1 = Image.fromarray(img1)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def histeq():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    img1 = cv.equalizeHist(img2)

    im1 = Image.fromarray(img1)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def meanfilt():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    blur = cv.blur(img2, (5, 5))

    im1 = Image.fromarray(blur)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def gaussian():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    blurg = cv.GaussianBlur(img2, (5, 5), 0)

    im1 = Image.fromarray(blurg)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def threhold():
    num = int(thresent.get())
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    ret, img1 = cv.threshold(img2, num, 255, cv.THRESH_BINARY)

    im1 = Image.fromarray(img1)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def med():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    median = cv.medianBlur(img2, 5)

    im1 = Image.fromarray(median)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def minf():
    img2 = Image.open(filepath)
    img2 = img2.resize((400, 200))
    img2 = img2.convert("L")
    min_filter = img2.filter(ImageFilter.MinFilter(size=3))

    im1 = min_filter
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def maxf():
    img2 = Image.open(filepath)
    img2 = img2.resize((400, 200))
    img2 = img2.convert("L")
    max_filter = img2.filter(ImageFilter.MaxFilter(size=3))

    im1 = max_filter
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def laplacian():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    laplacian_filter = cv.Laplacian(img2, cv.CV_64F, ksize=3)

    im1 = Image.fromarray(laplacian_filter)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def normlap():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    laplacian_filter = cv.Laplacian(img2, cv.CV_64F, ksize=3)
    laplacian_normalization = np.uint8(np.absolute(laplacian_filter))

    im1 = Image.fromarray(laplacian_normalization)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def lapOfGaus():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    blurg = cv.GaussianBlur(img2, (3, 3), 0)
    laplacian_filter = cv.Laplacian(blurg, cv.CV_64F, ksize=3)

    im1 = Image.fromarray(laplacian_filter)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def prewitt():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    blurg = cv.GaussianBlur(img2, (3, 3), 0)

    kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    img_prex = cv.filter2D(blurg, -1, kernelx)
    img_prey = cv.filter2D(blurg, -1, kernely)

    prewitt = img_prex + img_prey
    im1 = Image.fromarray(prewitt)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def sobel():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    sobelx = np.uint8(np.absolute(cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=3)))
    sobely = np.uint8(np.absolute(cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=3)))

    sobel = cv.bitwise_or(sobelx, sobely)
    im1 = Image.fromarray(sobel)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def canny():
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))
    blur = cv.GaussianBlur(img2, (5, 5), 0)

    canny = cv.Canny(blur, 10, 100)

    im1 = Image.fromarray(canny)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


def sharp():
    # 'laplacian', 'laplacian of gaussian', 'prewitt', 'sobel', 'Canny'
    typ = sharpening.get()
    img2 = cv.imread(filepath, 0)
    img2 = cv.resize(img2, (400, 200))

    if (typ == 'laplacian'):
        edge = cv.Laplacian(img2, cv.CV_64F, ksize=3)

    elif (typ == 'laplacian of gaussian'):
        blurg = cv.GaussianBlur(img2, (3, 3), 0)
        edge = cv.Laplacian(blurg, cv.CV_64F, ksize=3)

    elif (typ == 'prewitt'):
        blurg = cv.GaussianBlur(img2, (3, 3), 0)

        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        img_prex = cv.filter2D(blurg, -1, kernelx)
        img_prey = cv.filter2D(blurg, -1, kernely)

        edge = img_prex + img_prey

    elif (typ == 'sobel'):
        sobelx = np.uint8(np.absolute(
            cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=3)))
        sobely = np.uint8(np.absolute(
            cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=3)))

        edge = cv.bitwise_or(sobelx, sobely)

    else:
        blur = cv.GaussianBlur(img2, (5, 5), 0)
        edge = cv.Canny(blur, 10, 100)

    img2 = img2 + edge
    im1 = Image.fromarray(img2)
    imgtk1 = ImageTk.PhotoImage(image=im1)
    after.configure(image=imgtk1)
    after.image = imgtk1


root = Toplevel()
root.title("image processing")
f1 = Frame(root, bg="white")
f1.pack()
# photo viewing
photo_frame = LabelFrame(f1, bg="white")
photo_frame.grid(row=0, column=0, columnspan=2)

before = Label(photo_frame, text="before", bg="white")
after = Label(photo_frame, text="after", bg="white")
openfile = Button(photo_frame, text="Upload image",
                  command=openfile, bg="#C67B7B", fg="white")
before.grid(row=0, column=0)
after.grid(row=0, column=1)
openfile.grid(row=1, column=0, columnspan=2)

for widget in photo_frame.winfo_children():
    widget.grid_configure(padx=5)

# processing
process_frame = LabelFrame(f1, bg="white")
process_frame.grid(row=1, column=0)

identity = Button(process_frame, text="Original",
                  command=identity, bg="#C67B7B", fg="white", width=26)
identity.grid(row=0, column=0, columnspan=3)

negative = Button(process_frame, text="Negative ",
                  command=nagative, bg="#C67B7B", fg="white", width=26)
negative.grid(row=1, column=0)

log = Button(process_frame, text="Log transformation",
             command=logfun, bg="#C67B7B", fg="white", width=26)
logent = Entry(process_frame, bg="#F9D9D9", width=26)
log.grid(row=2, column=0)
logent.grid(row=2, column=1)

power = Button(process_frame, text="power-law transformation",
               command=powe, bg="#C67B7B", fg="white", width=26)
powent = Entry(process_frame, bg="#F9D9D9", width=26)
power.grid(row=3, column=0)
powent.grid(row=3, column=1)

thres = Button(process_frame, text="Threshold", command=threhold,
               bg="#C67B7B", fg="white", width=26)
thresent = Entry(process_frame, bg="#F9D9D9", width=26)
thres.grid(row=4, column=0)
thresent.grid(row=4, column=1)

strech = Button(process_frame, text="Contrast Stretching",
                command=contr, bg="#C67B7B", fg="white", width=26)
stechminent = Entry(process_frame, bg="#F9D9D9", width=26)
stechminent.insert(0, "min value")
strechmaxent = Entry(process_frame, bg="#F9D9D9", width=26)
strechmaxent.insert(0, "max value")
strech.grid(row=5, column=0)
stechminent.grid(row=5, column=1)
strechmaxent.grid(row=5, column=2)

slicing = Button(process_frame, text="Intensity Slicing",
                 command=intslic, bg="#C67B7B", fg="white", width=26)
minent = Entry(process_frame, bg="#F9D9D9", width=26)
minent.insert(0, "min value")
maxent = Entry(process_frame, bg="#F9D9D9", width=26)
maxent.insert(0, "max value")
slicing.grid(row=6, column=0)
minent.grid(row=6, column=1)
maxent.grid(row=6, column=2)


for widget in process_frame.winfo_children():
    widget.grid_configure(padx=5, pady=10)


sharp_frame = LabelFrame(f1, bg="white")
sharp_frame.grid(row=1, column=1)

meanf = Button(sharp_frame, text="Mean Filter",
               command=meanfilt, bg="#C67B7B", fg="white", width=26)
guasf = Button(sharp_frame, text="gaussian filter",
               command=gaussian, bg="#C67B7B", fg="white", width=26)
medianf = Button(sharp_frame, text="Median Filter",
                 command=med, bg="#C67B7B", fg="white", width=26)
meanf.grid(row=0, column=0)
guasf.grid(row=0, column=1)
medianf.grid(row=1, column=0)

mif = Button(sharp_frame, text="min filter", command=minf,
             bg="#C67B7B", fg="white", width=26)
mxf = Button(sharp_frame, text="max filter", command=maxf,
             bg="#C67B7B", fg="white", width=26)
lab = Button(sharp_frame, text="Laplacian filter",
             command=laplacian, bg="#C67B7B", fg="white", width=26)
mif.grid(row=1, column=1)
mxf.grid(row=2, column=0)
lab.grid(row=2, column=1)

lOg = Button(sharp_frame, text="laplacian of gaussian filter",
             command=lapOfGaus, bg="#C67B7B", fg="white", width=26)
pret = Button(sharp_frame, text="prewitt filter",
              command=prewitt, bg="#C67B7B", fg="white", width=26)
sob = Button(sharp_frame, text="Sobel filter",
             command=sobel, bg="#C67B7B", fg="white", width=26)
lOg.grid(row=3, column=0)
pret.grid(row=4, column=1)
sob.grid(row=4, column=0)


norlp = Button(sharp_frame, text="noramlized laplacian",
               command=normlap, bg="#C67B7B", fg="white", width=26)

norlp.grid(row=3, column=1)

can = Button(sharp_frame, text="Canny edge detection",
             command=canny, bg="#C67B7B", fg="white", width=26)

can.grid(row=5, column=0, columnspan=2)


sharpening = ttk.Combobox(sharp_frame, values=[
                          'laplacian', 'laplacian of gaussian', 'prewitt', 'sobel', 'Canny'])
sharpening.grid(row=6, column=1)

sharpened = Button(sharp_frame, text="Sharpen the photo",
                   command=sharp, bg="#C67B7B", fg="white", width=26)
sharpened.grid(row=6, column=0)

for widget in sharp_frame.winfo_children():
    widget.grid_configure(padx=5, pady=10)


root.mainloop()
