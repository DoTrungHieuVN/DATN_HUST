import sys
import time

from PyQt5.QtWidgets import QApplication, QMainWindow
from DATN_2 import Ui_MainWindow
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from PyQt5.QtCore import QThread ,pyqtSignal,Qt, QObject
from PyQt5 import QtGui
import socket
#robotstudio_ip = "127.0.0.1"
robotstudio_ip = "192.168.125.1"
robotstudio_port = 5000

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((robotstudio_ip, robotstudio_port))
    print("Đã kết nối thành công đến RobotStudio")
except socket.error as e:
    print("Lỗi kết nối:", e)
    exit()
class ServerListener(QObject):
    data_received = pyqtSignal(str)
    def __init__(self):
        super().__init__()
    def listen(self):
        while True:
            try:
                data_received=client_socket.recv(1024).decode()
                self.data_received.emit(data_received)
            except Exception as e:
                print("Lỗi khi nhận dữ liệu từ máy chủ:", e)
                break
class ServerListenerThread(QThread):
    def __init__(self, server_listener):
        super().__init__()
        self.server_listener = server_listener
    def run(self):
        self.server_listener.listen()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.thread = {}
        self.vuong = 0
        self.tron = 0
        self.x = 0
        self.y = 0
        self.goc = 0
        self.create_adjust_bar()
        self.uic.bt_start.clicked.connect(self.start_capture_video)
        self.uic.bt_stop.clicked.connect(self.stop_capture_video)
        self.uic.bt_reset.clicked.connect(self.reset_counters)
        self.server_listener = ServerListener()
        self.server_listener_thread = ServerListenerThread(self.server_listener)
        self.server_listener.data_received.connect(self.handle_received_data)
        self.server_listener_thread.start()

    def handle_received_data(self, data_received):
        if data_received == '0':
            self.tron+=1
        if data_received == '1':
            self.vuong+=1
        self.uic.label_vuong.setText(str(self.vuong))
        self.uic.label_tron.setText(str(self.tron))
        self.data_received =data_received

    def reset_counters(self):
        self.vuong=0
        self.tron=0
        self.x = 0
        self.y = 0
        self.goc = 0
        self.update_counters()
    def update_counters(self):
        self.uic.label_vuong.setText(str(self.vuong))
        self.uic.label_tron.setText(str(self.tron))
        self.uic.label_x.setText(str(self.x))
        self.uic.label_y.setText(str(self.y))
        self.uic.label_goc.setText(str(self.goc))
    def stop_capture_video(self):
        if 1 in self.thread:
            self.thread[1].stop()
            self.thread[1].wait()
        client_socket.close()

    def closeEvent(self, event):
        self.stop_capture_video()
    def start_capture_video(self):
        if 1 not in self.thread:
            self.thread[1] = capture_video(index=1,client_socket=client_socket)
            self.thread[1].start()
            self.thread[1].signal.connect(self.show_wedcam)
    def show_wedcam(self,img):
         qt_img = self.convert_qt(img)
         self.uic.camera_1.setPixmap(qt_img)

    def create_adjust_bar(self):
        cv2.namedWindow('AdjustBar')
        cv2.createTrackbar("L-H", "AdjustBar", 0, 100, self.nothing)
        cv2.createTrackbar("L-S", "AdjustBar", 0, 255, self.nothing)
        cv2.createTrackbar("L-V", "AdjustBar", 173, 255, self.nothing)
        cv2.createTrackbar("U-H", "AdjustBar", 167, 180, self.nothing)
        cv2.createTrackbar("U-S", "AdjustBar", 15, 255, self.nothing)
        cv2.createTrackbar("U-V", "AdjustBar", 255, 255, self.nothing)

    def nothing(self, x):
        pass
    def convert_qt(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cameraMatrix = np.array([[959.17315931, 0, 269.86695556], [0, 956.80204416, 221.05005383], [0, 0, 1]])
        distCoeffs = np.array([[-2.93004117e-01, 2.22589813e+00, -3.68278776e-03, 3.15708789e-03, -1.55876905e+01]])
        frame_undistorted = cv2.undistort(img, cameraMatrix, distCoeffs)
        hsv = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L-H", "AdjustBar")
        l_s = cv2.getTrackbarPos("L-S", "AdjustBar")
        l_v = cv2.getTrackbarPos("L-V", "AdjustBar")
        u_h = cv2.getTrackbarPos("U-H", "AdjustBar")
        u_s = cv2.getTrackbarPos("U-S", "AdjustBar")
        u_v = cv2.getTrackbarPos("U-V", "AdjustBar")
        i = 0
        j = 0
        lower_red = np.array([l_h, l_s, l_v])
        upper_red = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        center_list = np.empty((0, 2))
        orient_obtacles_draw = np.array([0, 0])
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if 2000 > area > 1200:
                sorted_corners = np.squeeze(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True))
                (cx, cy) = np.mean(sorted_corners, axis=0, dtype=float)
                new_center = np.array([cx, cy])
                center_list = np.vstack([center_list, new_center])
        s_max = 0
        i_max = 0
        if len(contours) > 0:
            if 2000 > area > 1200:
                cnt_max = contours[0]
                i = 0
                for point_i in center_list:
                    s = 0
                    for point_j in center_list:
                        s = s + np.linalg.norm(point_i - point_j)
                    if s > s_max:
                        s_max = s
                        cnt_max = contours[i]
                        i_max = i
                    i = i + 1
                orient_obtacles = np.array([1, 0])
                if len(center_list) == 2:
                    for i in range(len(center_list)):
                        if i != i_max:
                            orient_obtacles = center_list[i] - center_list[i_max]
                    orient_obtacles = orient_obtacles / np.linalg.norm(orient_obtacles)
                if (len(center_list) > 2):
                    if i_max == 0:
                        ui_boundary = center_list[1] - center_list[0]
                        uj_boundary = center_list[2] - center_list[0]
                        umax_i = ui_boundary
                        umax_j = uj_boundary
                    if i_max == 1:
                        ui_boundary = center_list[0] - center_list[1]
                        uj_boundary = center_list[2] - center_list[1]
                        umax_i = ui_boundary
                        umax_j = uj_boundary
                    if i_max > 1:
                        ui_boundary = center_list[0] - center_list[i_max]
                        uj_boundary = center_list[1] - center_list[i_max]
                        umax_i = ui_boundary
                        umax_j = uj_boundary
                    cos_alpha_max = abs(np.dot(ui_boundary, uj_boundary)) / (
                            np.linalg.norm(ui_boundary) * np.linalg.norm(uj_boundary))

                    for i in range(len(center_list)):
                        if i != i_max:
                            umax_i = center_list[i] - center_list[i_max]
                        for j in range(i + 1, len(center_list), 1):
                            if (j != i_max):
                                umax_j = center_list[j] - center_list[i_max]
                                cos_alpha = abs(np.dot(umax_i, umax_j)) / (
                                        np.linalg.norm(umax_i) * np.linalg.norm(umax_j))
                                if cos_alpha_max > cos_alpha:
                                    cos_alpha_max = cos_alpha
                                    ui_boundary = umax_i
                                    uj_boundary = umax_j
                    ui_bound_unit = ui_boundary / np.linalg.norm(ui_boundary)
                    uj_bound_unit = uj_boundary / np.linalg.norm(uj_boundary)
                    orient_obtacles = ui_bound_unit + uj_bound_unit
                    orient_obtacles = orient_obtacles / np.linalg.norm(orient_obtacles)
                orient_obtacles_draw = orient_obtacles * 100
                if len(center_list) > 0:
                    center_list = center_list.flatten()
                    point1 = (center_list[2 * i_max], center_list[2 * i_max + 1])
                    point1_draw = (int(center_list[2 * i_max]), int(center_list[2 * i_max + 1]))
                    point2 = (center_list[2 * i_max] + orient_obtacles_draw[0],
                              center_list[2 * i_max + 1] + orient_obtacles_draw[1])
                    point2_draw = (int(center_list[2 * i_max] + orient_obtacles_draw[0]),
                                   int(center_list[2 * i_max + 1] + orient_obtacles_draw[1]))
                    cv2.line(frame_undistorted, point1_draw, point2_draw, (50, 200, 255), 3)
                    sorted_corners_max = np.squeeze(
                        cv2.approxPolyDP(cnt_max, 0.02 * cv2.arcLength(cnt_max, True), True))
                    horizontal_vector = np.array([1, 0])
                    cv2.drawContours(frame_undistorted, [cnt_max], 0, (255, 0, 50), 2)
                    (cx_max, cy_max) = point1
                    self.x = "{:.2f}".format(cx_max)
                    self.y = "{:.2f}".format(cy_max)
                    cv2.circle(frame_undistorted, point1_draw, 7, (0, 0, 255), -1)
                    approx_max = cv2.approxPolyDP(cnt_max, 0.02 * cv2.arcLength(cnt_max, True), True)
                    if len(approx_max) == 4:
                        i = i + 1
                        cv2.putText(frame_undistorted, "Square", (point1_draw[0] + 15, point1_draw[1] + 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0))
                        cv2.putText(frame_undistorted, "X=", (point1_draw[0] - 22, point1_draw[1] - 22),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        cv2.putText(frame_undistorted, f'{point1[0]:.2f}', (point1_draw[0] + 12, point1_draw[1] - 22),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        cv2.putText(frame_undistorted, "Y=", (point1_draw[0] - 22, point1_draw[1] + 33),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        cv2.putText(frame_undistorted, f'{point1[1]:.2f}', (point1_draw[0] + 12, point1_draw[1] + 33),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        max_y_point = sorted_corners_max[np.argmax(sorted_corners_max[:, 1])]
                        min_x_point = sorted_corners_max[np.argmin(sorted_corners_max[:, 0])]
                        max_x_point = sorted_corners_max[np.argmax(sorted_corners_max[:, 0])]
                        min_y_point = sorted_corners_max[np.argmin(sorted_corners_max[:, 1])]
                        edge1_vector = max_y_point - min_x_point
                        edge2_vector = max_x_point - max_y_point
                        cos_beta1 = abs(np.dot(edge1_vector, orient_obtacles)) / (
                                np.linalg.norm(edge1_vector) * np.linalg.norm(orient_obtacles))
                        cos_beta2 = abs(np.dot(edge2_vector, orient_obtacles)) / (
                                np.linalg.norm(edge2_vector) * np.linalg.norm(orient_obtacles))
                        if cos_beta1 >= cos_beta2:
                            if np.dot(edge1_vector, orient_obtacles) > 0:
                                distance_vector = edge1_vector
                            else:
                                distance_vector = - edge1_vector
                            cos_theta = np.dot(distance_vector, horizontal_vector) / np.linalg.norm(
                                edge1_vector) * np.linalg.norm(
                                horizontal_vector)
                            cv2.circle(frame_undistorted, tuple(min_x_point), 5, (0, 0, 255), -1)
                            cv2.line(frame_undistorted, tuple(min_x_point), tuple(max_y_point), (0, 100, 255), 5)
                            cv2.line(frame_undistorted, tuple(min_y_point), tuple(max_x_point), (0, 100, 255), 5)
                        else:
                            if np.dot(edge2_vector, orient_obtacles) > 0:
                                distance_vector = edge2_vector
                            else:
                                distance_vector = - edge2_vector
                            cos_theta = np.dot(distance_vector, horizontal_vector) / np.linalg.norm(
                                edge2_vector) * np.linalg.norm(
                                horizontal_vector)
                            cv2.circle(frame_undistorted, tuple(max_y_point), 5, (0, 0, 255), -1)
                            cv2.line(frame_undistorted, tuple(max_y_point), tuple(max_x_point), (0, 100, 255), 5)
                            cv2.line(frame_undistorted, tuple(min_x_point), tuple(min_y_point), (0, 100, 255), 5)
                        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        self.goc = f'{angle:.2f}'
                        cv2.putText(frame_undistorted, f'Angle: {angle:.2f}',
                                    ((int(center_list[2 * i_max]) - 40, int(center_list[2 * i_max + 1]) - 40)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        distance_vector = distance_vector / np.linalg.norm(distance_vector)
                        distance_vector = distance_vector * 75
                        distance_point = point1 - distance_vector
                        cv2.circle(frame_undistorted, (int(distance_point[0]), int(distance_point[1])), 3,
                                   (255, 120, 120), 4)
                        cv2.putText(frame_undistorted, "X=", (int(distance_point[0]) - 22, int(distance_point[1]) - 16),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 120, 120), 1)
                        cv2.putText(frame_undistorted, f'{distance_point[0]:.2f}',
                                    (int(distance_point[0]) + 12, int(distance_point[1]) - 16),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (255, 120, 120), 1)
                        cv2.putText(frame_undistorted, "Y=", (int(distance_point[0]) - 22, int(distance_point[1]) + 23),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (255, 120, 120), 1)
                        cv2.putText(frame_undistorted, f'{distance_point[1]:.2f}',
                                    (int(distance_point[0]) + 12, int(distance_point[1]) + 23),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (255, 120, 120), 1)
                        cv2.line(frame_undistorted, (int(distance_point[0]), int(distance_point[1])), point1_draw,
                                 (255, 120, 120), 2)
                        if distance_point[1] > point1[1]:
                            angle = 90 - angle
                        else:
                            angle = angle - 90
                        kt = 1
                    elif len(approx_max) >= 8:
                        kt = 0
                        j = j + 1
                        cv2.putText(frame_undistorted, "Circle", (point1_draw[0] + 9, point1_draw[1] + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0))
                        cv2.putText(frame_undistorted, "X=", (point1_draw[0] - 22, point1_draw[1] - 22),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        cv2.putText(frame_undistorted, f'{point1[0]:.2f}', (point1_draw[0] + 12, point1_draw[1] - 22),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        cv2.putText(frame_undistorted, "Y=", (point1_draw[0] - 22, point1_draw[1] + 28),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        cv2.putText(frame_undistorted, f'{point1[1]:.2f}', (point1_draw[0] + 12, point1_draw[1] + 28),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        cos_gamma = np.dot(orient_obtacles, horizontal_vector) / (
                                np.linalg.norm(orient_obtacles) * np.linalg.norm(horizontal_vector))
                        angle = np.degrees(np.arccos(np.clip(cos_gamma, -1.0, 1.0)))
                        distance_vector = orient_obtacles
                        if np.dot(orient_obtacles, horizontal_vector) > 0:
                            angle = 90 - angle
                        else:
                            angle = - (angle - 90)
                        self.goc = f'{angle:.2f}'
                        cv2.putText(frame_undistorted, f'Angle: {angle:.2f} ',
                                    ((int(center_list[2 * i_max]) - 40, int(center_list[2 * i_max + 1]) - 40)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        distance_vector = distance_vector / np.linalg.norm(distance_vector)
                        distance_vector = distance_vector * 75
                        distance_point = point1 - distance_vector
                        cv2.circle(frame_undistorted, (int(distance_point[0]), int(distance_point[1])), 3,
                                   (255, 20, 100),
                                   4)
                        cv2.putText(frame_undistorted, "X=", (int(distance_point[0]) - 22, int(distance_point[1]) - 16),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 120, 120), 1)
                        cv2.putText(frame_undistorted, f'{distance_point[0]:.2f} ',
                                    (int(distance_point[0]) + 12, int(distance_point[1]) - 16),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5, (255, 120, 120), 1)
                        cv2.putText(frame_undistorted, "Y=", (int(distance_point[0]) - 22, int(distance_point[1]) + 23),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 120, 120), 1)
                        cv2.putText(frame_undistorted, f'{distance_point[1]:.2f} ',
                                    (int(distance_point[0]) + 12, int(distance_point[1]) + 23),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5, (255, 120, 120), 1)
                        cv2.line(frame_undistorted, (int(distance_point[0]), int(distance_point[1])), point1_draw,
                                 (255, 120, 120), 2)
                    if self.data_received == "Gui X":
                        client_socket.send(str("{:.2f}".format(cx_max)).encode())
                        self.uic.label_x.setText(str(self.x))
                    elif self.data_received == "Gui Y":
                        client_socket.send(str("{:.2f}".format(cy_max)).encode())
                        self.uic.label_y.setText(str(self.y))
                    elif self.data_received == "Gui Goc":
                        client_socket.send(str("{:.2f}".format(angle)).encode())
                        self.uic.label_goc.setText(str(self.goc))
                    elif self.data_received == "Gui kt":
                        client_socket.send(str(kt).encode())
                    elif self.data_received == "test":
                        client_socket.send(str(cx_max).encode())
                    elif self.data_received == "Gui X_dis":
                        client_socket.send(str("{:.2f}".format(distance_point[0])).encode())
                    elif self.data_received == "Gui Y_dis":
                        client_socket.send(str("{:.2f}".format(distance_point[1])).encode())
                self.data_received = ""
        h, w, ch = frame_undistorted.shape
        bytes_per_line = ch *w
        convert_to_Qt_format =QtGui.QImage(frame_undistorted.data,w,h,bytes_per_line,QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index=0,client_socket=None):
        self.index = index
        self.client_socket = client_socket
        print(" start threading", self.index)
        self.should_stop = False
        super(capture_video,self).__init__()
    def run(self):
        cap = cv2.VideoCapture(1)
        while not self.should_stop:
            ret, img = cap.read()
            img = img[126:370, 174:540]
            self.signal.emit(img)
            time.sleep(0.01)
        cap.release()
    def stop(self):
        print(" stop threading",self.index)
        self.should_stop = True
        self.terminate()
        self.wait()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())

