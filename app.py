import streamlit as st
from streamlit_cropper import st_cropper
import cv2
import numpy as np
import math
from PIL import Image

st.set_page_config(page_title="Cutoff Analyzer", layout="centered")

st.title("📱 모바일 컷오프라인 분석기")
st.caption("손가락으로 박스를 조절하여 분석 영역(ROI)을 설정하세요.")

# 1. 설정 영역
with st.sidebar:
    st.header("⚙️ 분석 설정")
    dist = st.number_input("스크린 거리 (mm)", value=10000, step=100)
    zoom_option = st.selectbox("촬영 배율", ["1x (Main)", "2x (Crop)", "5x (Telephoto)"], index=2)
    fov_map = {"1x (Main)": 53.1, "2x (Crop)": 28.2, "5x (Telephoto)": 11.2}
    fov = fov_map[zoom_option]

# 2. 이미지 업로드
col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Before 사진", type=['jpg', 'jpeg', 'png'])
with col2:
    after_file = st.file_uploader("After 사진", type=['jpg', 'jpeg', 'png'])

if before_file and after_file:
    img_b_pil = Image.open(before_file).convert("RGB")
    img_a_pil = Image.open(after_file).convert("RGB")
    
    st.divider()
    
    # 3. 손가락으로 ROI 조절 (Before 이미지 기준)
    st.subheader("🎯 분석 영역(ROI) 지정")
    st.info("아래 이미지의 사각형을 조절하여 컷오프 라인이 포함되게 하세요.")
    
    # 크롭 도구를 사용하여 ROI 좌표 추출 (aspect_ratio=None으로 자유 조절)
    roi_coords = st_cropper(img_b_pil, realtime_update=True, box_color='#FFE115', aspect_ratio=None)
    
    # 선택된 ROI 정보 (left, top, width, height)
    left, top, width, height = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
    
    if st.button("📊 분석 실행", use_container_width=True):
        # 넘파이 변환
        img_b = np.array(img_b_pil)
        img_a = np.array(img_a_pil)
        h, w = img_b.shape[:2]

        def detect_y(img, l, t, w_roi, h_roi):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            sob = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
            
            # ROI의 정중앙 X좌표 계산
            x_center = l + (w_roi // 2)
            # ROI Y범위 내에서 최대 에지 탐색
            roi_edge_data = sob[t : t + h_roi, x_center]
            return t + np.argmax(roi_edge_data), x_center

        y_b, x_c = detect_y(img_b, left, top, width, height)
        y_a, _ = detect_y(img_a, left, top, width, height)
        
        # 결과 계산
        p_diff = y_b - y_a
        deg = (p_diff * fov) / h
        mm = dist * math.tan(math.radians(deg))
        
        # 결과 표시
        res_color = "#FF4B4B" if deg > 0 else "#1C83E1"
        st.balloons()
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; border-left: 5px solid {res_color};">
            <h3 style="margin-top:0;">분석 결과</h3>
            <p>픽셀 변위: <b>{p_diff} px</b></p>
            <p>각도 변화: <span style='color:{res_color}; font-size:24px; font-weight:bold;'>{deg:+.4f}°</span></p>
            <p>수직 이동: <span style='color:{res_color}; font-size:24px; font-weight:bold;'>{mm:+.2f} mm</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 시각화 검증
        def draw_marker(img, x, y):
            res = img.copy()
            # 십자 마커 및 분석선
            cv2.line(res, (x, 0), (x, h), (255, 0, 0), 10)
            cv2.circle(res, (x, y), 25, (0, 255, 0), -1)
            return res

        st.image([draw_marker(img_b, x_c, y_b), draw_marker(img_a, x_c, y_a)], 
                 caption=["Before 분석점", "After 분석점"], use_column_width=True)
else:
    st.warning("분석을 시작하려면 사진 두 장을 업로드하세요.")
