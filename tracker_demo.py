import numpy as np
import cv2 as cv
import time
import matplotlib.pylab as plt

class MeanShift:

    def __init__(self, frame, track_window ):

        # ustavitveni pogoji
        self.maxiter = 10
        self.eps = 0
        self.bins = 16
        self.alpha = 0.0001

        # Analysis
        self.dt_avg_sum = 0
        self.iter_avg_sum = 0
        self.frames = 0

        # track_window dimenzije
        self.track_window = track_window
        (self.X0, self.Y0, self.width, self.height) = track_window  # X0 in Y0 sta zgornji levi kot ROI
        self.width = self.width+1
        self.height = self.height+1
        self.dx = np.floor(self.width / 2)
        self.dy = np.floor(self.height / 2)
        self.xs = np.round(np.array((self.X0 + self.dx+1, self.Y0 + self.dy+1)))  # center okvircka
        self.h = 2 * np.array((self.dx, self.dy)) + 1

        # sosedi
        self.Dx = np.arange(-self.dx, self.dx+1)
        self.Dy = np.arange(-self.dy, self.dy+1)
        self.X, self.Y = np.meshgrid(self.Dx, self.Dy)

        # Limite
        self.img_size = frame.shape
        self.xmin = self.dx + 1
        self.xmax = frame.shape[1] - self.xmin
        self.ymin = self.dy + 1
        self.ymax = frame.shape[0] - self.ymin

        # v primeru reinicializacije
        if self.xs[0] < self.xmin:
            self.xs[0] = self.xmin
        elif self.xs[0] > self.xmax:
            self.xs[0] = self.xmax

        if self.xs[1] < self.ymin:
            self.xs[1] = self.ymin
        elif self.xs[1] > self.ymax:
            self.xs[1] = self.ymax

        self.roi = frame[self.Y0:self.Y0 + self.height, self.X0:self.X0 + self.width]
        self.hsv_roi = cv.cvtColor(self.roi, cv.COLOR_BGR2HSV)
        # na tem mestu bi lahko za še boljše sledenje filtrirala hsv_roi in se s tem znebila robnih vrednosti, ki bi
        # samo motile izračun
        self.q = self.extract_hist(self.hsv_roi[:, :, 0], 179, self.width, self.height, 1)

    def update(self, frame):
        x = self.xs
        iter = 1

        t = time.time()
        while 1:
            print(x)
            Xi = self.X + x[0]
            Yi = self.Y + x[1]

            roi = frame[int((x[1]+self.Dy)[0]):(int((x[1]+self.Dy)[-1])+1), int((x[0]+self.Dx)[0]):(int((x[0]+self.Dx)[-1])+1),:]
            hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            p = self.extract_hist(hsv_roi[:, :, 0], 179, self.width, self.height, np.mean(self.h))

            v = np.sqrt(self.q / (p + 1e-10))

            # weight
            wi = self.BackProject(hsv_roi[:,:,0], v, self.width, self.height)


            #wi = cv.calcBackProject([hsv_roi],[0,1],v,[0,180,0,256],1)
            #print(wi.shape)
            xiwi = np.multiply(wi, Xi)
            yiwi = np.multiply(wi, Yi)
            xm = np.round(np.array((np.sum(xiwi), np.sum(yiwi))) / np.sum(wi),0)

            # v primeru, da dosezemo rob slike
            if xm[0] < self.xmin:
                xm[0] = self.xmin
            elif xm[0] > self.xmax:
                xm[0] = self.xmax

            if xm[1] < self.ymin:
                xm[1] = self.ymin
            elif xm[1] > self.ymax:
                xm[1] = self.ymax


            # preveri ustavitvene pogoje
            if (np.linalg.norm(xm - x) <= self.eps) or (iter >= self.maxiter):
                dt = time.time() - t
                break
            else:
                x = xm # posodobimo stanje
                iter += 1


        X0m = int(xm[0]-self.dx)
        Y0m = int(xm[1]-self.dy)


        self.X0 = X0m
        self.Y0 = Y0m

        self.q = (1 - self.alpha) * self.q + self.alpha * p

        self.xs = np.round(np.array((self.X0 + self.dx + 1, self.Y0 + self.dy + 1)))
        self.track_window = (X0m, Y0m, self.width, self.height)

        self.dt_avg_sum = self.dt_avg_sum + dt
        self.iter_avg_sum = self.iter_avg_sum + iter
        self.frames = self.frames + 1


        return (X0m, Y0m, self.width, self.height)


# POMOZNE FUNKCIJE

    def epanechnik_kernel(self, w, h, sigma):
        w2 = np.floor(w / 2)
        h2 = np.floor(h / 2)

        x = np.linspace(-int(w2), int(w2), int(w))
        y = np.linspace(-int(h2), int(h2), int(h))
        xv, yv = np.meshgrid(x, y)

        xv = np.true_divide(xv, np.amax(xv))
        yv = np.true_divide(yv,np.amax(yv))

        kernel = (np.ones((int(h), int(w))) - ((np.true_divide(xv,sigma))**2 + (np.true_divide(yv,sigma))**2))
        kernel = np.true_divide(kernel,np.amax(kernel))
        kernel[kernel < 0] = 0
        return kernel

    def extract_hist(self, ROI, bins, w, h, sigma):
        jedro = self.epanechnik_kernel(w, h, sigma)
        hist = np.zeros((bins+1, 1))
        for i in range(0, h):
            for j in range(0, w):
                hist[int(ROI[i, j]), 0] += 1 * jedro[i, j]
        return hist / np.sum(hist)

    def BackProject(self, roi, v, width, height):
        w = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                w[i, j] = v[int(roi[i, j]), 0]
        return w


#######################################################################################################################
class MeanShiftRGB:

    def __init__(self, frame, track_window):

        # ustavitveni pogoji
        self.maxiter = 10
        self.eps = 0
        self.bins = 16
        self.alpha = 0.0001

        # Analiza
        self.dt_avg_sum = 0
        self.iter_avg_sum = 0
        self.frames = 0

        # track_window dimenzije
        self.track_window = track_window
        (self.X0, self.Y0, self.width, self.height) = track_window  # X0 in Y0 sta zgornji levi kot ROI
        self.width = self.width+1
        self.height = self.height+1
        self.dx = np.floor(self.width / 2)
        self.dy = np.floor(self.height / 2)
        self.xs = np.round(np.array((self.X0 + self.dx+1, self.Y0 + self.dy+1)))  # center okvircka
        self.h = 2 * np.array((self.dx, self.dy)) + 1

        # sosedi
        self.Dx = np.arange(-self.dx, self.dx+1)
        self.Dy = np.arange(-self.dy, self.dy+1)
        self.X, self.Y = np.meshgrid(self.Dx, self.Dy)

        # Limite
        self.img_size = frame.shape
        self.xmin = self.dx + 1
        self.xmax = frame.shape[1] - self.xmin
        self.ymin = self.dy + 1
        self.ymax = frame.shape[0] - self.ymin

        # v primeru reinicializacije
        if self.xs[0] < self.xmin:
            self.xs[0] = self.xmin
        elif self.xs[0] > self.xmax:
            self.xs[0] = self.xmax

        if self.xs[1] < self.ymin:
            self.xs[1] = self.ymin
        elif self.xs[1] > self.ymax:
            self.xs[1] = self.ymax

        self.roi = frame[self.Y0:self.Y0 + self.height, self.X0:self.X0 + self.width]
        self.q = self.extract_hist(self.roi, 255, self.width, self.height, 1)

    def update(self, frame):
        x = self.xs
        iter = 1

        t = time.time()
        while 1:
            Xii = self.X + x[0]
            s = np.shape(Xii)
            Xi = np.zeros((s[0],s[1],3))
            Xi[:,:,0] = Xii[:,:]
            Xi[:, :, 1] = Xii[:, :]
            Xi[:, :, 2] = Xii[:, :]
            Yii = self.Y + x[1]
            Yi = np.zeros((s[0], s[1], 3))
            Yi[:, :, 0] = Yii[:, :]
            Yi[:, :, 1] = Yii[:, :]
            Yi[:, :, 2] = Yii[:, :]

            #hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            roi = frame[int((x[1]+self.Dy)[0]):(int((x[1]+self.Dy)[-1])+1), int((x[0]+self.Dx)[0]):(int((x[0]+self.Dx)[-1])+1),:]
            p = self.extract_hist(roi, 255, self.width, self.height, np.mean(self.h))

            v = np.sqrt(self.q / (p + 1e-10))

            wi = self.BackProject(roi, v, self.width, self.height)

            xiwi = np.multiply(wi, Xi)
            yiwi = np.multiply(wi, Yi)
            xm = np.round(np.array((np.sum(xiwi), np.sum(yiwi))) / np.sum(wi),0)

            # v primeru, da dosezemo rob slike
            if xm[0] < self.xmin:
                xm[0] = self.xmin
            elif xm[0] > self.xmax:
                xm[0] = self.xmax

            if xm[1] < self.ymin:
                xm[1] = self.ymin
            elif xm[1] > self.ymax:
                xm[1] = self.ymax

            # preveri ustavitvene pogoje
            if (np.linalg.norm(xm - x) <= self.eps) or (iter >= self.maxiter):
                dt = time.time() - t
                break
            else:
                x = xm # posodobimo stanje
                iter += 1

        X0m = int(xm[0]-self.dx)
        Y0m = int(xm[1]-self.dy)

        self.X0 = X0m
        self.Y0 = Y0m

        self.q = (1 - self.alpha) * self.q + self.alpha * p

        self.xs = np.round(np.array((self.X0 + self.dx + 1, self.Y0 + self.dy + 1)))
        self.track_window = (X0m, Y0m, self.width, self.height)

        self.dt_avg_sum = self.dt_avg_sum + dt
        self.iter_avg_sum = self.iter_avg_sum + iter
        self.frames = self.frames + 1

        return self.track_window


# POMOZNE FUNKCIJE

    def epanechnik_kernel(self, w, h, sigma):
        w2 = np.floor(w / 2)
        h2 = np.floor(h / 2)

        x = np.linspace(-int(w2), int(w2), int(w))
        y = np.linspace(-int(h2), int(h2), int(h))
        xv, yv = np.meshgrid(x, y)

        xv = np.true_divide(xv, np.amax(xv))
        yv = np.true_divide(yv,np.amax(yv))

        kernel = (np.ones((int(h), int(w))) - ((np.true_divide(xv,sigma))**2 + (np.true_divide(yv,sigma))**2))
        kernel = np.true_divide(kernel,np.amax(kernel))
        kernel[kernel < 0] = 0
        return kernel

    def extract_hist(self, ROI, bins, w, h, sigma):
        jedro = self.epanechnik_kernel(w, h, sigma)
        hist = np.zeros((bins+1, 1))
        for i in range(0, h):
            for j in range(0, w):
                for k in range(0,3):
                    #print(i, j, k)
                    hist[int(ROI[i, j, k]), 0] += 1 * jedro[i, j]
        return hist / np.sum(hist)

    def BackProject(self, roi, v, width, height):
        w = np.zeros((height, width, 3))
        for i in range(0, height):
            for j in range(0, width):
                for k in range(0,3):
                    w[i, j, k] = v[int(roi[i, j, k]), 0]
        return w

class CAMshift:

    def __init__(self, frame, track_window):

        # ustavitveni pogoji
        self.maxiter = 10
        self.eps = 0
        self.bins = 16
        self.alpha = 0.0001

        # Analysis
        self.dt_avg_sum = 0
        self.iter_avg_sum = 0
        self.frames = 0

        # track_window dimenzije
        self.track_window = track_window
        (self.X0, self.Y0, self.width, self.height) = track_window  # X0 in Y0 sta zgornji levi kot ROI
        self.width = self.width + 1
        self.height = self.height + 1
        self.dx = np.floor(self.width / 2)
        self.dy = np.floor(self.height / 2)
        self.xs = np.round(np.array((self.X0 + self.dx + 1, self.Y0 + self.dy + 1)))  # center okvircka
        self.h = 2 * np.array((self.dx, self.dy)) + 1
        self.h_CAM = 2 * np.array((self.dx + 5, self.dy + 5)) + 1

        # sosedi
        self.Dx = np.arange(-self.dx, self.dx + 1)
        self.Dy = np.arange(-self.dy, self.dy + 1)
        self.X, self.Y = np.meshgrid(self.Dx, self.Dy)

        # Limite
        self.img_size = frame.shape
        self.xmin = self.dx + 1
        self.xmax = frame.shape[1] - self.xmin
        self.ymin = self.dy + 1
        self.ymax = frame.shape[0] - self.ymin

        # v primeru reinicializacije
        if self.xs[0] < self.xmin:
            self.xs[0] = self.xmin
        elif self.xs[0] > self.xmax:
            self.xs[0] = self.xmax

        if self.xs[1] < self.ymin:
            self.xs[1] = self.ymin
        elif self.xs[1] > self.ymax:
            self.xs[1] = self.ymax

        self.roi = frame[self.Y0:self.Y0 + self.height, self.X0:self.X0 + self.width]
        self.hsv_roi = cv.cvtColor(self.roi, cv.COLOR_BGR2HSV)
        self.q = self.extract_hist(self.hsv_roi[:, :, 0], 179, self.width, self.height, 1)

        self.roi_CAM = frame[(self.Y0-5):(self.Y0 + self.height+5), (self.X0-5):(self.X0 + self.width+5)]
        self.hsv_roi_CAM = cv.cvtColor(self.roi_CAM, cv.COLOR_BGR2HSV)
        self.q_CAM = self.extract_hist(self.hsv_roi_CAM[:, :, 0], 179, self.width+10, self.height+10, 1)

    def update(self, frame):
        x = self.xs
        iter = 1

        t = time.time()
        while 1:
            Xi = self.X + x[0]
            Yi = self.Y + x[1]

            # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            roi = frame[int((x[1] + self.Dy)[0]):(int((x[1] + self.Dy)[-1]) + 1),
                  int((x[0] + self.Dx)[0]):(int((x[0] + self.Dx)[-1]) + 1), :]
            hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            p = self.extract_hist(hsv_roi[:, :, 0], 179, self.width, self.height, np.mean(self.h))

            roi_CAM = frame[int((x[1] + self.Dy)[0]-5):(int((x[1] + self.Dy)[-1]+5) + 1),
                  int((x[0] + self.Dx)[0]-5):(int((x[0] + self.Dx)[-1]+5) + 1), :]
            hsv_roi_CAM = cv.cvtColor(roi_CAM, cv.COLOR_BGR2HSV)
            p_CAM = self.extract_hist(hsv_roi_CAM[:, :, 0], 179, self.width+10, self.height+10, np.mean(self.h))

            v = np.sqrt(self.q / (p + 1e-10))
            v_CAM = np.sqrt(self.q_CAM / (p_CAM + 1e-10))

            wi = self.BackProject(hsv_roi[:, :, 0], v, self.width, self.height)
            wi_CAM = self.BackProject(hsv_roi_CAM[:, :, 0], v_CAM, self.width+10, self.height+10)

            print(wi.shape)
            print(Xi.shape)
            xiwi = np.multiply(wi, Xi)
            yiwi = np.multiply(wi, Yi)
            xm = np.round(np.array((np.sum(xiwi), np.sum(yiwi))) / np.sum(wi), 0)

            # v primeru, da dosezemo rob slike
            if xm[0] < self.xmin:
                xm[0] = self.xmin
            elif xm[0] > self.xmax:
                xm[0] = self.xmax

            if xm[1] < self.ymin:
                xm[1] = self.ymin
            elif xm[1] > self.ymax:
                xm[1] = self.ymax

            # preveri ustavitvene pogoje
            if (np.linalg.norm(xm - x) <= self.eps) or (iter >= self.maxiter):
                dt = time.time() - t
                break
            else:
                x = xm  # posodobimo stanje
                iter += 1

        X0m = int(xm[0] - self.dx)
        Y0m = int(xm[1] - self.dy)
        # konec meanshift-a


        momenti = cv.moments(wi)
        M00 = momenti["m00"]
        mi_11 = momenti["mu11"]
        mi_20 = momenti["mu20"]
        mi_02 = momenti["mu02"]

        theta = np.arctan2(2 * mi_11, mi_20 - mi_02 + np.sqrt(4 * (mi_11) ** 2 + (mi_20 - mi_02) ** 2))

        Imax = mi_20 * (np.cos(theta)) ** 2 + 2 * mi_11 * np.sin(theta) * np.cos(theta) + mi_02 * (np.sin(theta)) ** 2
        Imin = mi_20 * (np.sin(theta)) ** 2 - 2 * mi_11 * np.sin(theta) * np.cos(theta) + mi_02 * (np.cos(theta)) ** 2

        dolga_os = 4 * np.sqrt(Imax / M00)
        kratka_os = 4*np.sqrt(Imin/M00)

        self.width = int(np.round(np.max(np.array((dolga_os*np.cos(theta), kratka_os*np.sin(theta))))))
        self.height = int(np.round(np.max(np.array((dolga_os*np.sin(theta), kratka_os*np.cos(theta))))))

        self.X0 = X0m
        self.Y0 = Y0m

        self.dx = np.floor(self.width / 2)
        self.dy = np.floor(self.height / 2)
        self.xs = np.round(np.array((self.X0 + self.dx + 1, self.Y0 + self.dy + 1)))  # center okvircka
        self.h = 2 * np.array((self.dx, self.dy)) + 1
        self.h_CAM = 2 * np.array((self.dx + 5, self.dy + 5)) + 1

        # sosedi
        self.Dx = np.arange(-self.dx, self.dx + 1)
        self.Dy = np.arange(-self.dy, self.dy + 1)
        self.X, self.Y = np.meshgrid(self.Dx, self.Dy)

        self.q = (1 - self.alpha) * self.q + self.alpha * p

        self.xs = np.round(np.array((self.X0 + self.dx + 1, self.Y0 + self.dy + 1)))
        self.track_window = (X0m, Y0m, self.width, self.height)

        self.dt_avg_sum = self.dt_avg_sum + dt
        self.iter_avg_sum = self.iter_avg_sum + iter
        self.frames = self.frames + 1

        self.img_size = frame.shape
        self.xmin = self.dx + 1
        self.xmax = frame.shape[1] - self.xmin
        self.ymin = self.dy + 1
        self.ymax = frame.shape[0] - self.ymin

        # v primeru reinicializacije
        if self.xs[0] < self.xmin:
            self.xs[0] = self.xmin
        elif self.xs[0] > self.xmax:
            self.xs[0] = self.xmax

        if self.xs[1] < self.ymin:
            self.xs[1] = self.ymin
        elif self.xs[1] > self.ymax:
            self.xs[1] = self.ymax

        self.roi = frame[self.Y0:self.Y0 + self.height, self.X0:self.X0 + self.width]
        self.hsv_roi = cv.cvtColor(self.roi, cv.COLOR_BGR2HSV)
        self.q = self.extract_hist(self.hsv_roi[:, :, 0], 179, self.width, self.height, 1)

        self.roi_CAM = frame[(self.Y0 - 5):(self.Y0 + self.height + 5), (self.X0 - 5):(self.X0 + self.width + 5)]
        self.hsv_roi_CAM = cv.cvtColor(self.roi_CAM, cv.COLOR_BGR2HSV)
        self.q_CAM = self.extract_hist(self.hsv_roi_CAM[:, :, 0], 179, self.width + 10, self.height + 10, 1)

        return (X0m, Y0m, self.width, self.height)

    # POMOZNE FUNKCIJE

    def epanechnik_kernel(self, w, h, sigma):
        w2 = np.floor(w / 2)
        h2 = np.floor(h / 2)

        x = np.linspace(-int(w2), int(w2), int(w))
        y = np.linspace(-int(h2), int(h2), int(h))
        xv, yv = np.meshgrid(x, y)

        xv = np.true_divide(xv, np.amax(xv))
        yv = np.true_divide(yv, np.amax(yv))

        kernel = (np.ones((int(h), int(w))) - ((np.true_divide(xv, sigma)) ** 2 + (np.true_divide(yv, sigma)) ** 2))
        kernel = np.true_divide(kernel, np.amax(kernel))
        kernel[kernel < 0] = 0
        return kernel

    def extract_hist(self, ROI, bins, w, h, sigma):
        jedro = self.epanechnik_kernel(w, h, sigma)
        hist = np.zeros((bins + 1, 1))
        for i in range(0, h):
            for j in range(0, w):
                hist[int(ROI[i, j]), 0] += 1 * jedro[i, j]
        return hist / np.sum(hist)

    def BackProject(self, roi, v, width, height):
        w = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                w[i, j] = v[int(roi[i, j]), 0]
        return w



############################################################################################
#Izbira algoritma
# Komentar: dela samo MeanShift_HSV. MeanShift tudi načeloma dela, vendar veliko prepočasi


use_mean_shift_HSV = 1
use_mean_shift_RGB = 0
use_CAMshift = 0


# INICIALIZACIJA

# izbira vide-a in pripadajoči track_window
cap = cv.VideoCapture('zogica.mp4')
#cap = cv.VideoCapture('cars.mp4')
#cap = cv.VideoCapture('boy-walking.mp4')
ret, frame = cap.read()

#plt.imshow(frame)
#plt.show()



# cars
#X0, Y0, width, height = 620, 5, 30, 20

#zogica
X0, Y0, width, height = 700, 400, 80, 80

#boy_walking
#X0,Y0,width, height = 45,145,30,40


track_window = (X0,Y0,width,height)



if use_mean_shift_HSV == 1:
    ms = MeanShift(frame, track_window)
elif use_mean_shift_RGB == 1:
    ms = MeanShiftRGB(frame, track_window)
elif use_CAMshift == 1:
    ms = CAMshift(frame, track_window)


############################
# dejanski zagon programa

while (1):

    ret ,frame = cap.read()
    if ret == True:
        track_window = ms.update(frame)
        # Draw it on image
        x, y, ms.width, ms.height = track_window
        print(x, y, ms.width, ms.height)
        okvircek = cv.rectangle(frame, (x, y), (x + ms.width, y + ms.height), 255, 2)
        cv.imshow('okvircek', okvircek)
        k = cv.waitKey(10) & 0xff


    else:
        break
cv.destroyAllWindows()
cap.release()

