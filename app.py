# 파일명: app.py
import streamlit as st
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

# 페이지 기본 설정
st.set_page_config(page_title="Headlamp Cut-off Analyzer", layout="wide")

# 🚨 [광학 스펙 정밀 교정] 아이폰 15 프로 맥스 전용 하드웨어 광학 데이터베이스 (수직 FOV 기준)
# 1x 메인(24mm), 2x 크롭(48mm), 5x 망원(120mm 테트라프리즘) 사양 완벽 반영
IPHONE_15_PRO_MAX_SPECS = {
    "1x (Main - 24mm)": 57.3,
    "2x (In-Sensor Crop - 48mm)": 30.3,
    "5x (Telephoto - 120mm)": 12.4
}

# 제어 패널 UI (사이드바)
st.sidebar.title("🎛️ 제어 패널")
D = st.sidebar.number_input("거리 (mm)", value=10000, step=500)
gain = st.sidebar.slider("밝기 Gain", 0.5, 3.0, 1.0, step=0.1)
manual_offset = st.sidebar.number_input("수동 Offset (mm)", value=0.0, step=0.1)

# 📱 촬영 기기 사양 고정 및 배율 메뉴 개편
st.sidebar.markdown("---")
st.sidebar.subheader("📱 촬영 기기 사양")
st.sidebar.info("기기: iPhone 15 Pro Max")

# 사용자가 직관적으로 선택할 수 있도록 1배, 2배, 5배 광학 스펙만 바인딩
zoom_labels = list(IPHONE_15_PRO_MAX_SPECS.keys())
zoom_sel = st.sidebar.selectbox("촬영 배율 선택", zoom_labels)
FOV = IPHONE_15_PRO_MAX_SPECS[zoom_sel] # 선택한 배율의 정밀 수직 FOV 적용

st.title("🔦 Headlamp Cut-off Analyzer")
st.caption("iPhone 15 Pro Max 광학 렌즈 및 하드웨어 센서 스펙 맞춤형 정밀 분석기")

# 이미지 업로더
uploaded_file = st.file_uploader("헤드램프 조사 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # PIL 이미지로 로드 후 아이폰 EXIF 태그 기준 자동 회전 정정
    raw_pil = Image.open(uploaded_file)
    origin_pil = ImageOps.exif_transpose(raw_pil)
    W_img, H_img = origin_pil.size
    
    st.subheader("🔍 분석 영역(ROI) 설정")
    
    # 영속적 세션 상태 공간 정의 및 초기화 보장
    if "roi_x1" not in st.session_state:
        st.session_state.roi_x1 = int(W_img * 0.35)
        st.session_state.roi_y1 = int(H_img * 0.35)
        st.session_state.roi_x2 = int(W_img * 0.65)
        st.session_state.roi_y2 = int(H_img * 0.65)
    
    st.info("💡 이미지 위의 노란색 테두리 상자를 손가락으로 드래그하거나 모서리를 잡고 늘려보세요.")
    
    # 터치 크롭 도구 실행
    cropped_box = st_cropper(
        origin_pil, 
        realtime_update=True, 
        box_color='#ffdd00', 
        aspect_ratio=None,
        return_type='box',
        key="headlamp_cropper"
    )
    
    # 반환되는 딕셔너리의 Key 이름(x/left, width/w) 호환성 연동
    if cropped_box and isinstance(cropped_box, dict):
        tx1 = int(cropped_box.get('x', cropped_box.get('left', st.session_state.roi_x1)))
        ty1 = int(cropped_box.get('y', cropped_box.get('top', st.session_state.roi_y1)))
        
        tw = int(cropped_box.get('width', cropped_box.get('w', st.session_state.roi_x2 - st.session_state.roi_x1)))
        th = int(cropped_box.get('height', cropped_box.get('h', st.session_state.roi_y2 - st.session_state.roi_y1)))
        
        tx2 = tx1 + tw
        ty2 = ty1 + th
        
        if tw >= 10 and th >= 10:
            if (st.session_state.roi_x1 != tx1 or 
                st.session_state.roi_y1 != ty1 or 
                st.session_state.roi_x2 != tx2 or 
                st.session_state.roi_y2 != ty2):
                
                st.session_state.roi_x1 = tx1
                st.session_state.roi_y1 = ty1
                st.session_state.roi_x2 = tx2
                st.session_state.roi_y2 = ty2
                st.rerun()

    # 연산부 변수 할당
    x1, y1, x2, y2 = st.session_state.roi_x1, st.session_state.roi_y1, st.session_state.roi_x2, st.session_state.roi_y2

    # OpenCV 프로세싱 진행
    img_b = cv2.cvtColor(np.array(origin_pil), cv2.COLOR_RGB2BGR)
    roi_b = img_b[y1:y2, x1:x2]
    
    # 핵심 알고리즘 가동
    roi_g = np.clip(roi_b.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(roi_g, cv2.COLOR_BGR2HSV)
    
    m = cv2.add(
        cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255])), 
        cv2.add(
            cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255])), 
            cv2.add(
                cv2.inRange(hsv, np.array([35,80,80]), np.array([90,255,255])), 
                cv2.inRange(hsv, np.array([25,50,70]), np.array([35,255,255]))
            )
        )
    )
    
    if np.max(np.sum(m, axis=1)) > 50*255:
        l_y = np.argmax(np.sum(m, axis=1)) + y1
        
        # 🚨 정밀 교정된 FOV 데이터 기반 픽셀당 도수(Degree) 연산 실행
        deg_per_px = FOV / H_img
        std_y = l_y + (math.degrees(math.atan(0.01)) / deg_per_px)
        
        gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(cv2.GaussianBlur(gray, (9,9), 0), cv2.CV_64F, 0, 1, ksize=5)
        rel_l_y = l_y - y1
        grad[max(0, rel_l_y-15):min(roi_b.shape[0], rel_l_y+15), :] = 0
        
        c_y_auto = np.argmax(np.mean(grad, axis=1)) + y1
        c_y = c_y_auto - (math.degrees(math.atan(manual_offset / D)) / deg_per_px)
        
        mm_raw = D * math.tan(math.radians((l_y - c_y) * deg_per_px))
        mm_std = D * math.tan(math.radians((std_y - c_y) * deg_per_px))
        pct_raw, pct_std = (mm_raw / D) * 100, (mm_std / D) * 100
        
        st.markdown("---")
        
        # 결과 시각화 레이아웃 (반응형 2칼럼 배치)
        view_col, graph_col = st.columns([1, 1])
        
        with view_col:
            st.subheader("🖼️ 분석 결과 이미지")
            disp_img = img_b.copy()
            cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 221, 255), 3)
            cv2.line(disp_img, (x1, int(l_y)), (x2, int(l_y)), (0, 0, 255), 3)
            cv2.line(disp_img, (x1, int(c_y)), (x2, int(c_y)), (0, 255, 0), 4)
            
            disp_img_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            st.image(disp_img_rgb, use_container_width=True, caption="[레드: 레이저선 / 그린: 컷오프선]")
            
        with graph_col:
            st.subheader("📊 Intensity Profile")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#121212')
            ax.set_facecolor('#1e1e1e')
            
            profile = np.mean(gray, axis=1)
            y_idx = np.arange(y1, y2)
            mm_ax = [D * math.tan(math.radians((l_y - y) * deg_per_px)) for y in y_idx]
            
            ax.plot(mm_ax, profile, color='#ffffff', lw=2)
            y_max = np.max(profile) if len(profile) > 0 else 255
            ax.axvline(x=0, color='#ff3b30', lw=1.5)
            ax.axvline(x=-(D*0.01), color='#0a84ff', linestyle='--')
            ax.axvline(x=mm_raw, color='#30d158', lw=2)
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.grid(True, color='#555', lw=0.8)
            ax.set_xlabel("Height (mm)", color='white')
            ax.set_ylabel("Brightness", color='white')
            ax.tick_params(colors='white')
            st.pyplot(fig)
            
        st.markdown("---")
        status_color = '#0a84ff' if mm_std > 0 else '#ff453a'
        status_text = '높음 (UP)' if mm_std > 0 else '낮음 (DOWN)'
        
        res_html = f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 3px solid {status_color}; text-align: center;">
            <p style="margin: 0; color: #888; font-size: 14px;">[북미 사양 기준]</p>
            <p style="margin: 5px 0 15px 0; font-size: 24px; font-weight: bold; color: white;">{mm_raw:+.1f} mm ({pct_raw:+.2f}%)</p>
            <p style="margin: 0; color: #888; font-size: 14px;">[유럽 사양 기준]</p>
            <p style="margin: 5px 0 15px 0; font-size: 24px; font-weight: bold; color: white;">{mm_std:+.1f} mm ({pct_std:+.2f}%)</p>
            <p style="margin: 0; font-size: 22px; font-weight: bold; color: {status_color};">상태: {status_text}</p>
        </div>
        """
        st.html(res_html)
    else:
        st.error("레이저 라인을 인식하지 못했습니다. 밝기 Gain을 조절하거나 다른 이미지를 시도해 주세요.")
