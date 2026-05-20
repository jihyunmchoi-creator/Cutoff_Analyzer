# 파일명: app.py
import streamlit as st
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper
import streamlit.components.v1 as components

# 페이지 기본 설정
st.set_page_config(page_title="Headlamp Cut-off Analyzer", layout="wide")

# html2canvas를 이용한 클라이언트 사이드 영역별 캡처 스크립트 정의
# 특정 ID 요소를 찾아 이미지화한 뒤 다운로드 링크를 강제 트리거합니다.
SCREENSHOT_JS = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
function captureElement(elementId, filename) {
    // Streamlit 상위 DOM에서 해당 ID를 가진 요소를 탐색
    const element = window.parent.document.getElementById(elementId);
    if (!element) {
        alert("캡처할 영역을 찾을 수 없습니다. 분석을 먼저 실행해주세요.");
        return;
    }
    
    // 모바일 대응 및 화질 향상을 위해 scale 옵션 부여
    html2canvas(element, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#121212"
    }).then(canvas => {
        const link = window.parent.document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        link.click();
    }).catch(err => {
        console.error("Screenshot error:", err);
    });
}
</script>
"""

# 자바스크립트 함수 베이스 로드
components.html(SCREENSHOT_JS, height=0, width=0)

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

# 🚨 [신규 기능] 사이드바 스크린샷 저장 버튼 세션
st.sidebar.markdown("---")
st.sidebar.subheader("📸 결과 스크린샷 저장")

# 각 버튼 클릭 시 부모창의 자바스크립트 함수를 실행하도록 구성
if st.sidebar.button("🖼️ 분석 이미지 저장", use_container_width=True):
    components.html("<script>captureElement('capture-view', 'analysis_image.png');</script>", height=0, width=0)

if st.sidebar.button("📊 그래프 영역 저장", use_container_width=True):
    components.html("<script>captureElement('capture-graph', 'intensity_profile.png');</script>", height=0, width=0)

if st.sidebar.button("📋 판정 결과 저장", use_container_width=True):
    components.html("<script>captureElement('capture-result', 'judgment_result.png');</script>", height=0, width=0)


# 제목 및 캡션 텍스트
st.markdown("# 🔦 Headlamp<br>Cut-off Analyzer", unsafe_allow_html=True)
st.caption("모바일 시인성 개선. 유럽/북미 사양별 위치 판정 분석")

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

    # 알고리즘: 원본 이미지 기반 vivid 컬러 마스킹 및 분석
    img_b = cv2.cvtColor(np.array(origin_pil), cv2.COLOR_RGB2BGR)
    orig_hsv = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    
    # vivid 녹색 및 적색 범위 마스킹
    vivid_mask = cv2.add(
        cv2.inRange(orig_hsv, np.array([35, 120, 100]), np.array([90, 255, 255])),
        cv2.add(
            cv2.inRange(orig_hsv, np.array([0, 150, 100]), np.array([10, 255, 255])),
            cv2.inRange(orig_hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
        )
    )
    roi_vivid_mask = vivid_mask[y1:y2, x1:x2]
    vivid_px_ratio = (cv2.countNonZero(roi_vivid_mask) / (roi_vivid_mask.shape[0] * roi_vivid_mask.shape[1])) * 100
    
    if vivid_px_ratio > 0.05: 
        vivid_y_sums = np.sum(roi_vivid_mask, axis=1)
        c_y_auto = np.argmax(vivid_y_sums) + y1
        deg_per_px = FOV / H_img
        c_y = c_y_auto - (math.degrees(math.atan(manual_offset / D)) / deg_per_px)

        # 레이저 주변 영역 마스킹 (검색 제외 영역 30px 유지)
        safe_margin = 30 
        roi_b = img_b[y1:y2, x1:x2]
        roi_g = np.clip(roi_b.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(cv2.GaussianBlur(gray, (9,9), 0), cv2.CV_64F, 0, 1, ksize=5)
        rel_c_y = int(c_y_auto - y1)
        grad[max(0, rel_c_y - safe_margin):min(roi_b.shape[0], rel_c_y + safe_margin), :] = 0
        l_y = np.argmax(np.mean(grad, axis=1)) + y1
        
        # 수치 계산
        std_y = c_y + (math.degrees(math.atan(0.01)) / deg_per_px)
        mm_raw = D * math.tan(math.radians((c_y - l_y) * deg_per_px))
        mm_std = D * math.tan(math.radians((std_y - l_y) * deg_per_px))
        pct_raw, pct_std = (mm_raw / D) * 100, (mm_std / D) * 100
        
        st.markdown("---")
        view_col, graph_col = st.columns([1, 1])
        
        with view_col:
            # 🚨 html2canvas 추적용 div 매핑
            st.markdown('<div id="capture-view">', unsafe_allow_html=True)
            st.subheader("🖼️ 분석 결과 이미지")
            disp_img = img_b.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 3.0 
            f_thick = 7

            # 1. 레이저 정렬선 (빨간색)
            cv2.line(disp_img, (0, int(c_y)), (W_img, int(c_y)), (0, 0, 255), 5)
            cv2.putText(disp_img, "LASER LINE", (50, int(c_y) - 30), font, f_scale, (0, 0, 255), f_thick)

            # 2. 인식된 컷오프 라인 (초록색)
            cv2.line(disp_img, (0, int(l_y)), (W_img, int(l_y)), (0, 255, 0), 5)
            cv2.putText(disp_img, "CUT-OFF", (50, int(l_y) - 30), font, f_scale, (0, 255, 0), f_thick)

            # 3. 유럽 사양 -1% 하향 가상 라인 (파란색)
            cv2.line(disp_img, (0, int(std_y)), (W_img, int(std_y)), (255, 0, 0), 4)
            cv2.putText(disp_img, "EU -1% Line", (W_img - 850, int(std_y) + 100), font, f_scale, (255, 0, 0), f_thick)

            # ROI 박스 (하늘색)
            cv2.rectangle(disp_img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            
            disp_img_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            st.image(disp_img_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with graph_col:
            # 🚨 html2canvas 추적용 div 매핑
            st.markdown('<div id="capture-graph">', unsafe_allow_html=True)
            st.subheader("📊 Intensity Profile")
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#121212')
            ax.set_facecolor('#1e1e1e')
            
            profile = np.mean(gray, axis=1)
            y_idx = np.arange(y1, y2)
            mm_ax = [D * math.tan(math.radians((c_y - y) * deg_per_px)) for y in y_idx]
            
            ax.plot(mm_ax, profile, color='#ffffff', lw=2, label="Profile")
            
            ax.axvline(x=0, color='#ff3b30', lw=2, label="Laser Line (Ref)")
            ax.axvline(x=mm_raw, color='#30d158', lw=2.5, label="Cut-off Line")
            ax.axvline(x=-(D*0.01), color='#0a84ff', linestyle='--', lw=2, label="EU -1% Line")
            
            ax.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='white', labelcolor='white', fontsize='medium')
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.grid(True, color='#555', lw=0.8)
            ax.set_xlabel("Height (mm)", color='white')
            ax.set_ylabel("Brightness", color='white')
            
            ax.tick_params(colors='white')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 각 기준선 대비 순수 위치 판정(UP/DOWN)
        if abs(mm_raw) <= 5.0:
            us_status, us_color = "정상 (OK)", "#30d158"
        elif mm_raw > 5.0:
            us_status, us_color = "높음 (UP)", "#ff453a"
        else:
            us_status, us_color = "낮음 (DOWN)", "#ff9f0a"

        if abs(mm_std) <= 5.0: 
            eu_status, eu_color = "정상 (OK)", "#0a84ff"
        elif mm_std > 5.0:
            eu_status, eu_color = "높음 (UP)", "#ff453a"
        else:
            eu_status, eu_color = "낮음 (DOWN)", "#ff9f0a"
        
        # 🚨 html2canvas 추적용 div 매핑
        res_html = f"""
        <div id="capture-result" style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 1px solid #444; text-align: left;">
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
        st.error("레이저 라인을 인식하지 못했습니다. vivid 컬러의 라인을 ROI 내에 포함시키거나, 밝기 Gain을 조절해 주세요.")
