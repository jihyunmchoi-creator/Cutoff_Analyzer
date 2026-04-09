import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

st.set_page_config(page_title="헤드램프 분석기", layout="centered")

st.title("🔦 헤드램프 변위 분석기")
st.caption("아이폰 사진을 업로드하거나 즉시 촬영하여 분석하세요.")

# 설정 사이드바
with st.sidebar:
    st.header("⚙️ 분석 설정")
    dist = st.number_input("스크린 거리 (mm)", value=10000, step=100)
    zoom_option = st.selectbox("촬영 배율", ["1x (Main)", "2x (Crop)", "5x (Telephoto)"], index=2)
    fov_map = {"1x (Main)": 53.1, "2x (Crop)": 28.2, "5x (Telephoto)": 11.2}
    fov = fov_map[zoom_option]

# 이미지 업로드 (카메라 촬영 가능)
col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Before 사진 (촬영/선택)", type=['jpg', 'jpeg', 'png'], key="before")
with col2:
    after_file = st.file_uploader("After 사진 (촬영/선택)", type=['jpg', 'jpeg', 'png'], key="after")

if before_file and after_file:
    # 이미지 로드
    img_b = np.array(Image.open(before_file))
    img_a = np.array(Image.open(after_file))
    
    # 해상도 정보
    h, w = img_b.shape[:2]
    
    st.divider()
    
    # 분석 위치 및 ROI 설정 (슬라이더)
    st.subheader("🎯 분석 영역 설정")
    x_pos = st.slider("분석 X좌표 (가로 위치)", 0, w, w // 2)
    roi_y = st.slider("에지 탐색 Y범위 (상하)", 0, h, (int(h*0.2), int(h*0.8)))
    
    if st.button("📊 분석 실행", use_container_width=True):
        def detect_y(img, x, r_min, r_max):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            sob = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
            roi_data = sob[r_min:r_max, x]
            return r_min + np.argmax(roi_data)

        y_b = detect_y(img_b, x_pos, roi_y[0], roi_y[1])
        y_a = detect_y(img_a, x_pos, roi_y[0], roi_y[1])
        
        # 결과 계산
        p_diff = y_b - y_a
        deg = (p_diff * fov) / h
        mm = dist * math.tan(math.radians(deg))
        
        # 결과 표시
        res_color = "red" if deg > 0 else "blue"
        st.markdown(f"""
        ### 분석 결과
        - **픽셀 변위:** `{p_diff} px`
        - **각도 변화:** <span style='color:{res_color}; font-size:20px; font-weight:bold;'>{deg:+.4f}°</span>
        - **수직 이동:** <span style='color:{res_color}; font-size:20px; font-weight:bold;'>{mm:+.2f} mm</span>
        """, unsafe_allow_html=True)
        
        # 결과 이미지 시각화
        def draw_result(img, x, y, r):
            res_img = img.copy()
            cv2.line(res_img, (x, 0), (x, h), (255, 0, 0), 5) # X축
            cv2.line(res_img, (x-50, y), (x+50, y), (0, 255, 0), 10) # 검출 Y
            # ROI 표시
            overlay = res_img.copy()
            cv2.rectangle(overlay, (0, 0), (w, r[0]), (0,0,0), -1)
            cv2.rectangle(overlay, (0, r[1]), (w, h), (0,0,0), -1)
            return cv2.addWeighted(overlay, 0.3, res_img, 0.7, 0)

        st.image([draw_result(img_b, x_pos, y_b, roi_y), 
                  draw_result(img_a, x_pos, y_a, roi_y)], 
                 caption=["Before 분석 지점", "After 분석 지점"], use_column_width=True)