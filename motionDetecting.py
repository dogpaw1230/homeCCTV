from datetime import datetime

import cv2
import numpy as np
import os

# 감도 설정(카메라 품질에 따라 조정 필요)
thresh = 25  # 달라진 픽셀 값 기준치 설정
max_diff = 5  # 달라진 픽셀 갯수 기준치 설정

# 카메라 캡션 장치 준비
a, b, c = None, None, None
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # 프레임 폭을 480으로 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)  # 프레임 높이를 320으로 설정

# 움직임 감지 중 녹화 상태를 추적
is_record = False
on_record = False
out = None

# 움직임 감지 녹화 설정
save_dir = 'C:/dogpawProjects/motionRecord/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 인코딩 포맷 문자
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (int(width), int(height))  # 프레임 크기

cnt_record = 0      # 영상 녹화 시간 관련 변수
max_cnt_record = 900  # 최소 촬영시간

# 영상 무한 루프
if cap.isOpened():
    ret, a = cap.read()  # a 프레임 읽기
    ret, b = cap.read()  # b 프레임 읽기

    while ret:
        ret, c = cap.read()  # c 프레임 읽기
        draw = c.copy()  # 출력 영상에 사용할 복제본
        if not ret:
            break

        # 저장할 영상 파일 이름 설정
        now = datetime.now()
        file_name = now.strftime("%Y_%m_%d_%H_%M_%S")

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

        # 두 차이에 대해서 AND 연산, 두 영상의 차이가 모두 발견된 경우
        diff = cv2.bitwise_and(diff1_t, diff2_t)

        # 열림 연산으로 노이즈 제거 ---①
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

        # 차이가 발생한 픽셀의 갯수 판단 후 사각형 그리기
        diff_cnt = cv2.countNonZero(diff)
        if diff_cnt > max_diff:
            is_record = True
            if not on_record:
                file_path = os.path.join(save_dir, f'{file_name}.avi')
                out = cv2.VideoWriter(file_path, fourcc, fps, size)
                print(f"Recording started: {file_path}")
                cnt_record = max_cnt_record
            nzero = np.nonzero(diff)  # 0이 아닌 픽셀의 좌표 얻기(y[...], x[...])
            cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])),
                          (max(nzero[1]), max(nzero[0])), (0, 255, 0), 2)
            cv2.putText(draw, "Motion Detected", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            # print("start: " + str(cnt_record))
        if is_record:
            on_record = True
            out.write(draw)
            cnt_record -= 1
            # print("녹화중: " + str(cnt_record))
        if cnt_record == 0:
            is_record = False
            on_record = False

        # 컬러 스케일 영상과 스레시홀드 영상을 통합해서 출력
        stacked = np.hstack((draw, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('motion sensor', stacked)

        # 다음 비교를 위해 영상 순서 정리
        a = b
        b = c

        if cv2.waitKey(1) & 0xFF == 27:
            break
