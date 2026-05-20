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

# 아이폰 15 프로 맥스 전용 하드웨어 광학 데이터베이스 (수직 FOV 기준)
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

st.sidebar.markdown("---")
st.sidebar.subheader("📱 촬영 기기 사양")
st.sidebar.info("기기: iPhone 15 Pro Max")

zoom_labels = list(IPHONE_15_PRO_MAX_SPECS.keys())
zoom_sel = st.sidebar.selectbox("촬영 배율 선택", zoom_labels)
FOV = IPHONE_15_PRO_MAX_SPECS[zoom_sel]

st.title("🔦 Headlamp Cut-off Analyzer")
st.caption("북미/유럽 사양별 합부 판정이 각각 독립적으로 분리 표시되는 정밀 분석기")

# 이미지 업로더
uploaded_file = st.file_uploader("헤드램프 조사 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_pil = Image.open(uploaded_file)
    origin_pil = ImageOps.exif_transpose(raw_pil)
    W_img, H_img = origin_pil.size
    
    st.subheader("🔍 분석 영역(ROI) 설정")
    
    if "roi_x1" not in st.session_state:
        st.session_state.roi_x1 = int(W_img * 0.35)
        st.session_state.roi_y1 = int(H_img * 0.35)
        st.session_state.roi_x2 = int(W_img * 0.65)
        st.session_state.roi_y2 = int(H_img * 0.65)
    
    st.info("💡 이미지 위의 노란색 테두리 상자를 손가락으로 드래그하거나 모서리를 잡고 늘려보세요.")
    
    cropped_box = st_cropper(
        origin_pil, 
        realtime_update=True, 
        box_color='#ffdd00', 
        aspect_ratio=None,
        return_type='box',
        key="headlamp_cropper"
    )
    
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

    x1, y1, x2, y2 = st.session_state.roi_x1, st.session_state.roi_y1, st.session_state.roi_x2, st.session_state.roi_y2

    img_b = cv2.cvtColor(np.array(origin_pil), cv2.COLOR_RGB2BGR)
    roi_b = img_b[y1:y2, x1:x2]
    
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
        
        # 🚨 [해결 핵심 로직] 북미/유럽 사양별 독립적 판정 알고리즘 분리 구현
        # 1. 북미 사양 기준 판정 (컷오프선이 레이저선 정렬선[0mm] 대비 어떤지 판정)
        if abs(mm_raw) <= 5.0: # 임의의 오차 허용값 예시 (필요시 조정 가능)
            us_status = "정상 (OK)"
            us_color = "#30d158" # 녹색
        elif mm_raw > 5.0:
            us_status = "높음 (UP)"
            us_color = "#ff453a" # 적색
        else:
            us_status = "낮음 (DOWN)"
            us_color = "#ff9f0a" # 주황색

        # 2. 유럽 사양 기준 판정 (컷오프선이 하향 1% 정렬선[-D*0.01 mm] 대비 어떤지 판정)
        # mm_std가 0보다 크면 1%선보다 위로 올라간 것이고, 작으면 아래로 내려간 것임
        if abs(mm_std) <= 5.0: 
            eu_status = "정상 (OK)"
            eu_color = "#0a84ff" # 청색
        elif mm_std > 5.0:
            eu_status = "높음 (UP)"
            eu_color = "#ff453a" # 적색
        else:
            eu_status = "낮음 (DOWN)"
            eu_color = "#ff9f0a" # 주황색
        
        # UI 레이아웃 화면에 개별 렌더링
        res_html = f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 1px solid #444; text-align: left;">
            <div style="margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #333;">
                <span style="color: #888; font-size: 14px; font-weight: bold; display: block; margin-bottom: 5px;">🇺🇸 [북미 사양 기준 결과]</span>
                <span style="font-size: 26px; font-weight: bold; color: white; display: inline-block; margin-right: 20px;">{mm_raw:+.1f} mm ({pct_raw:+.2f}%)</span>
                <span style="font-size: 20px; font-weight: bold; color: {us_color};">상태: {us_status}</span>
            </div>
            <div>
                <span style="color: #888; font-size: 14px; font-weight: bold; display: block; margin-bottom: 5px;">🇪🇺 [유럽 사양 기준 결과]</span>
                <span style="font-size: 26px; font-weight: bold; color: white; display: inline-block; margin-right: 20px;">{mm_std:+.1f} mm ({pct_std:+.2f}%)</span>
                <span style="font-size: 20px; font-weight: bold; color: {eu_color};">상태: {eu_status}</span>
            </div>
        </div>
        """
        st.html(res_html)
    else:
        st.error("레이저 라인을 인식하지 못했습니다. 밝기 Gain을 조절하거나 다른 이미지를 시도해 주세요.")
