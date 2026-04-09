import streamlit as st
from streamlit_cropper import st_cropper
import cv2
import numpy as np
import math
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="헤드램프 ROI 분석기", layout="centered")

st.title("📱 모바일 헤드램프 분석기")
st.caption("아이폰에서 사진을 촬영하거나 업로드하여 컷오프 변화량을 분석하세요.")

# 1. 사이드바 설정 영역
with st.sidebar:
    st.header("⚙️ 분석 설정")
    dist = st.number_input("스크린 거리 (mm)", value=10000, step=100)
    zoom_option = st.selectbox("촬영 배율", ["1x (Main)", "2x (Crop)", "5x (Telephoto)"], index=2)
    
    # iPhone 15 Pro Max 기준 화각(FOV) 데이터
    fov_map = {"1x (Main)": 53.1, "2x (Crop)": 28.2, "5x (Telephoto)": 11.2}
    fov = fov_map[zoom_option]
    st.info(f"선택된 배율의 수직 화각: {fov}°")

# 2. 이미지 업로드 영역
col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Before 사진", type=['jpg', 'jpeg', 'png'])
with col2:
    after_file = st.file_uploader("After 사진", type=['jpg', 'jpeg', 'png'])

if before_file and after_file:
    # PIL 이미지로 로드
    img_b_pil = Image.open(before_file).convert("RGB")
    img_a_pil = Image.open(after_file).convert("RGB")
    
    st.divider()
    
    # 3. 마우스/터치 ROI 조절 (streamlit-cropper 사용)
    st.subheader("🎯 분석 영역(ROI) 지정")
    st.info("아래 사진의 사각형을 움직여 '컷오프 라인'이 포함되게 하세요. 사각형의 좌우 중앙이 분석 기준선이 됩니다.")
    
    # 박스 조절 도구 (TypeError 방지를 위해 리턴값 안전하게 처리)
    roi_coords = st_cropper(img_b_pil, realtime_update=True, box_color='#FFE115', aspect_ratio=None)
    
    # 좌표 안전 추출 및 정수화
    if roi_coords:
        left = int(roi_coords.get('left', 0))
        top = int(roi_coords.get('top', 0))
        width = int(roi_coords.get('width', img_b_pil.width))
        height = int(roi_coords.get('height', img_b_pil.height))
    else:
        left, top, width, height = 0, 0, img_b_pil.width, img_b_pil.height

    # 4. 분석 실행 버튼
    if st.button("📊 분석 실행", use_container_width=True):
        try:
            # OpenCV 처리를 위한 넘파이 변환
            img_b = np.array(img_b_pil)
            img_a = np.array(img_a_pil)
            h, w = img_b.shape[:2]

            # 분석 지점(X center) 및 범위 확정
            x_center = int(left + (width // 2))
            y_min, y_max = int(top), int(top + height)

            def detect_y(img, x, r_min, r_max):
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # 노이즈 제거를 위한 가우시안 블러
                blur = cv2.GaussianBlur(gray, (15, 15), 0)
                # 수직 에지 강조 (Sobel)
                sob = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
                
                # 지정된 ROI Y범위 내에서만 탐색
                roi_edge_data = sob[r_min:r_max, x]
                if len(roi_edge_data) == 0:
                    return r_min
                return r_min + np.argmax(roi_edge_data)

            # Before/After 각각 에지 검출
            y_b = detect_y(img_b, x_center, y_min, y_max)
            y_a = detect_y(img_a, x_center, y_min, y_max)
            
            # 물리량 계산
            p_diff = y_b - y_a
            deg_diff = (p_diff * fov) / h
            h_diff = dist * math.tan(math.radians(deg_diff))
            
            # 5. 결과 시각화
            res_color = "#FF4B4B" if deg_diff > 0 else "#1C83E1"
            st.balloons()
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; border-left: 5px solid {res_color};">
                <h3 style="margin:0; color:#31333F;">분석 결과</h3>
                <p style="margin:10px 0 5px 0;">픽셀 변위: <b>{p_diff} px</b></p>
                <p style="margin:0;">각도 변화: <span style='color:{res_color}; font-size:24px; font-weight:bold;'>{deg_diff:+.4f}°</span></p>
                <p style="margin:0;">수직 이동: <span style='color:{res_color}; font-size:24px; font-weight:bold;'>{h_diff:+.2f} mm</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # 검증 이미지 생성 함수
            def get_debug_img(img, x, y):
                # 원본 복사 후 마커 그리기
                canvas = img.copy()
                # 분석 수직선 (파란색)
                cv2.line(canvas, (x, 0), (x, h), (255, 0, 0), 8)
                # 검출된 포인트 (초록색 점)
                cv2.circle(canvas, (x, y), 30, (0, 255, 0), -1)
                return canvas

            st.image([get_debug_img(img_b, x_center, y_b), 
                      get_debug_img(img_a, x_center, y_a)], 
                     caption=["Before 분석 위치", "After 분석 위치"], use_column_width=True)

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
else:
    st.info("분석을 위해 Before와 After 사진을 업로드해 주세요.")
