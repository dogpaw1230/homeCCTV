from datetime import datetime
import cv2
import numpy as np
import os
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# 감도 설정(카메라 품질에 따라 조정 필요)
thresh = 25
max_diff = 5

# 카메라 캡션 장치 준비
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# 움직임 감지 녹화 설정
save_dir = 'C:/dogpawProjects/motionRecord/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4x')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (int(width), int(height))

# MJPEG 스트리밍을 위한 프레임 생성 함수
def generate_frames():
    a, b, c = None, None, None
    is_record = False
    on_record = False
    out = None
    cnt_record = 0
    max_cnt_record = 900

    if cap.isOpened():
        ret, a = cap.read()
        ret, b = cap.read()

        while ret:
            ret, c = cap.read()
            draw = c.copy()
            if not ret:
                break

            # 3개의 영상을 그레이 스케일로 변경
            a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            # a-b, b-c 절대 값 차 구하기
            diff1 = cv2.absdiff(a_gray, b_gray)
            diff2 = cv2.absdiff(b_gray, c_gray)

            # 스레시홀드로 기준치 이내의 차이는 무시
            ret, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
            ret, diff2_t = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)

            # 두 차이에 대해서 AND 연산
            diff = cv2.bitwise_and(diff1_t, diff2_t)

            # 차이가 발생한 픽셀의 갯수 판단 후 사각형 그리기
            diff_cnt = cv2.countNonZero(diff)
            if diff_cnt > max_diff:
                is_record = True
                if not on_record:
                    now = datetime.now()
                    file_name = now.strftime("%Y_%m_%d_%H_%M_%S")
                    file_path = os.path.join(save_dir, f'{file_name}.mp4')
                    out = cv2.VideoWriter(file_path, fourcc, fps, size)
                    cnt_record = max_cnt_record
                nzero = np.nonzero(diff)
                cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])),
                              (max(nzero[1]), max(nzero[0])), (0, 255, 0), 2)
                cv2.putText(draw, "Motion Detected", (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

            if is_record:
                on_record = True
                out.write(draw)
                cnt_record -= 1
            if cnt_record == 0:
                is_record = False
                on_record = False

            # draw 프레임을 클라이언트로 전송
            ret, buffer = cv2.imencode('.jpg', draw)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            a = b
            b = c

# 웹 페이지에서 실시간 스트리밍 엔드포인트
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket을 이용한 캡처 요청 처리
@socketio.on('capture')
def handle_capture():
    now = datetime.now()
    file_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    capture_path = os.path.join(save_dir, f'{file_name}_capture.jpg')

    # 현재 프레임 캡처
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(capture_path, frame)
        socketio.emit('capture_response', {'image_url': capture_path})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
