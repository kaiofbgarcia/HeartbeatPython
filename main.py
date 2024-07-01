import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image

import cv2
import datetime
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

cascade_path = 'C:/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"Error loading cascade classifier from {cascade_path}")
    exit(1)

class MainApp(App):
    def build(self):
        self.start_time = None
        self.firstFrame = None
        self.time_list = []
        self.R = []
        self.G = []
        self.B = []
        self.pca = FastICA(n_components=3)
        self.frame_num = 0

        self.layout = BoxLayout(orientation='vertical')
        self.header = Label(text="Heartbeat Monitor", size_hint=(1, 0.1),
                            color=(1, 1, 1, 1), font_size='20sp', bold=True)
        
        self.image = Image(size_hint=(1, 0.8))
        self.label = Label(text="Starting...", size_hint=(1, 0.1),
                           color=(0, 0, 0, 1), font_size='20sp')
        
        self.footer = Label(text="by Baby Smartcare", size_hint=(1, 0.1),
                            color=(1, 1, 1, 1), font_size='16sp', bold=True)

        self.label = Label(text="Starting...")
        self.layout.add_widget(self.header)
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.label)
        self.layout.add_widget(self.footer)
        

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.label.text = "Failed to open webcam"
            return self.layout

        return self.layout

    def update(self, dt):
        ret, frame = self.cap.read()
        if ret:
            self.frame_num += 1
            if self.firstFrame is None:
                self.start_time = datetime.datetime.now()
                self.time_list.append(0)
                self.firstFrame = frame
                old_gray = cv2.cvtColor(self.firstFrame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(old_gray, 1.3, 5)
                if len(faces) == 0:
                    self.firstFrame = None
                else:
                    for (x, y, w, h) in faces:
                        x2 = x + w
                        y2 = y + h
                        cv2.rectangle(self.firstFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        VJ_mask = np.zeros_like(self.firstFrame)
                        VJ_mask = cv2.rectangle(VJ_mask, (x, y), (x + w, y + h), (255, 0, 0), -1)
                        VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                    ROI = VJ_mask
                    ROI_color = cv2.bitwise_and(ROI, ROI, mask=VJ_mask)
                    R_new, G_new, B_new, _ = cv2.mean(ROI_color, mask=ROI)
                    self.R.append(R_new)
                    self.G.append(G_new)
                    self.B.append(B_new)

            else:
                current = datetime.datetime.now() - self.start_time
                current = current.total_seconds()
                self.time_list.append(current)
                ROI_color = cv2.bitwise_and(frame, frame)
                R_new, G_new, B_new, _ = cv2.mean(ROI_color)
                self.R.append(R_new)
                self.G.append(G_new)
                self.B.append(B_new)
                if self.frame_num >= 900:
                    N = 900
                    G_std = StandardScaler().fit_transform(np.array(self.G[-(N-1):]).reshape(-1, 1))
                    G_std = G_std.reshape(1, -1)[0]
                    R_std = StandardScaler().fit_transform(np.array(self.R[-(N-1):]).reshape(-1, 1))
                    R_std = R_std.reshape(1, -1)[0]
                    B_std = StandardScaler().fit_transform(np.array(self.B[-(N-1):]).reshape(-1, 1))
                    B_std = B_std.reshape(1, -1)[0]
                    T = 1 / (len(self.time_list[-(N-1):]) / (self.time_list[-1] - self.time_list[-(N-1)]))
                    X_f = self.pca.fit_transform(np.array([R_std, G_std, B_std]).transpose()).transpose()
                    N = len(X_f[0])
                    yf = fft(X_f[1])
                    yf = yf / np.sqrt(N)
                    xf = fftfreq(N, T)
                    xf = fftshift(xf)
                    yplot = fftshift(abs(yf))
                    
                    fft_plot = yplot
                    fft_plot[xf <= 0.75] = 0

                    bpm = xf[fft_plot[xf <= 4].argmax()] * 60
                    self.label.text = f'{bpm:.2f} bpm'
                    
                    # Comentar a parte que exibe o gráfico
                    # plt.figure(1)
                    # plt.gcf().clear()
                    # plt.plot(xf[(xf >= 0) & (xf <= 4)], fft_plot[(xf >= 0) & (xf <= 4)])
                    # plt.pause(0.0001)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def on_stop(self):
        self.cap.release()

if __name__ == '__main__':
    MainApp().run()
