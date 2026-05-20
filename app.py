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

# 기기 FOV 데이터베이스 정의
DEVICE_FOV_DB = {
    "아이폰 15 프로 맥스": [58.0, 31.0, 12.5],
    "아이폰 16 프로 맥스": [58.0, 31.0, 12.5],
    "아이폰 17 프로 맥스": [58.0, 31.0, 11.0],
    "갤럭시 S25 울트라": [58.1, 30.5, 11.2, 5.8],
    "갤럭시 S26 울트라": [58.1, 30.5, 11.2, 5.8]
}

# 제어 패널 UI (사이드바)
st.sidebar.title("🎛️ 제어 패널")
D = st.sidebar.number_input("거리 (mm)", value=10000, step=500)
gain = st.sidebar.slider("밝기 Gain", 0.5, 3.0, 1.0, step=0.1)
manual_offset = st.sidebar.number_input("수동 Offset (mm)", value=0.0, step=0.1)

device_sel = st.sidebar.selectbox("기기 선택", list(DEVICE_FOV_DB.keys()))
fov_options = DEVICE_FOV_DB[device_sel]
zoom_labels = ["1x (Main)", "2x (Crop)", "3x (Tele)", "5x (Tele)", "10x (Tele)"]
zoom_sel = st.sidebar.selectbox(
    "배율 선택", 
    range(len(fov_options)), 
    format_func=lambda x: zoom_labels[x] if x < len(zoom_labels) else f"{x+1}x"
)
FOV = fov_options[zoom_sel]

st.title("🔦 Headlamp Cut-off Analyzer")
st.caption("터치 좌표 실시간 연동 안정화 루틴이 적용된 모바일 분석기")

# 이미지 업로더
uploaded_file = st.file_uploader("헤드램프 조사 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # PIL 이미지로 로드 후 아이폰 EXIF 태그 기준 자동 회전 정정
    raw_pil = Image.open(uploaded_file)
    origin_pil = ImageOps.exif_transpose(raw_pil)
    W_img, H_img = origin_pil.size
    
    st.subheader("🔍 분석 영역(ROI) 설정")
    
    # 1. 영속적 세션 상태 공간 정의 및 초기화 보장
    if "roi_x1" not in st.session_state:
        st.session_state.roi_x1 = int(W_img * 0.35)
        st.session_state.roi_y1 = int(H_img * 0.35)
        st.session_state.roi_x2 = int(W_img * 0.65)
        st.session_state.roi_y2 = int(H_img * 0.65)
    
    st.info("💡 이미지 위의 노란색 테두리 상자를 손가락으로 드래그하거나 모서리를 잡고 늘려보세요.")
    
    # 2. 터치 크롭 도구 실행
    cropped_box = st_cropper(
        origin_pil, 
        realtime_update=True, 
        box_color='#ffdd00', 
        aspect_ratio=None,
        return_type='box'
    )
    
    # 3. 🚨 실시간 감지 연동 코어 로직
    # 손가락을 움직여 새로운 상자 정보가 들어오면 즉시 세션 내부 영구 좌표를 덮어씁니다.
    if cropped_box is not None and isinstance(cropped_box, dict) and 'x' in cropped_box:
        tx1 = int(cropped_box['x'])
        ty1 = int(cropped_box['y'])
        tx2 = int(tx1 + cropped_box['w'])
        ty2 = int(ty1 + cropped_box['h'])
        
        # 최소 크기(10픽셀) 이상 움직였을 때만 상태를 즉각 강제 변경하여 분석란에 전송
        if (tx2 - tx1) >= 10 and (ty2 - ty1) >= 10:
            st.session_state.roi_x1 = tx1
            st.session_state.roi_y1 = ty1
            st.session_state.roi_x2 = tx2
            st.session_state.roi_y2 = ty2

    # 최종 연산부에서 사용할 좌표는 무조건 세션에 락인된 값을 실시간 추적하도록 강제 고정
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
