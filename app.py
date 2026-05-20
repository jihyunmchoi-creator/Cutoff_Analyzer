# 파일명: app.py
import streamlit as st
import cv2
import numpy as np
import math
import io
import base64  
import streamlit.components.v1 as components  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

# 페이지 기본 설정
st.set_page_config(page_title="Headlamp Cut-off Analyzer", layout="wide")

# iOS 크로스오리진(CORS) 및 페이지 유실을 방지하는 안전 다운로드 함수
def ios_safe_download_button(label, data, file_name, mime_type):
    b64 = base64.b64encode(data).decode()
    custom_html = f"""
    <a href="data:{mime_type};base64,{b64}" download="{file_name}" target="_blank" style="
        display: block;
        width: 100%;
        background-color: #262730;
        color: #ffffff;
        padding: 0.45rem 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(250, 250, 250, 0.2);
        font-size: 14px;
        font-weight: 500;
        text-decoration: none;
        box-sizing: border-box;
        text-align: center;
        margin-bottom: 8px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        transition: background-color 0.1s ease-in-out;
    " onmouseover="this.style.backgroundColor='#3e404f'" onmouseout="this.style.backgroundColor='#262730'">
        {label}
    </a>
    """
    with st.sidebar:
        components.html(custom_html, height=42)

# 아이폰 15 프로 맥스 전용 하드웨어 광학 데이터베이스 (수직 FOV 기준)
IPHONE_15_PRO_MAX_SPECS = {
    "1x (Main - 24mm)": 57.3,
    "2x (In-Sensor Crop - 48mm)": 30.3,
    "5x (Telephoto - 120mm)": 12.4
}

# 제어 패널 UI (사이드바 기본 레이아웃)
st.sidebar.title("🎛️ 제어 패널")

# 🚨 [핵심 추가] 모바일 상하 스크롤 안정화를 위한 ROI 가동 잠금 토글 스위치
st.sidebar.subheader("🔒 모바일 스크롤 잠금 해제")
roi_edit_mode = st.sidebar.toggle("🔍 ROI 영역 수정 모드 활성화", value=False)

if not roi_edit_mode:
    st.sidebar.caption("💡 수정 모드가 꺼져 있을 때는 이미지 영역을 드래그해도 화면 스크롤이 자유롭습니다.")
else:
    st.sidebar.caption("⚠️ 수정 중에는 모바일 스크롤이 제한될 수 있습니다. 조절 후 스위치를 꺼주세요.")

st.sidebar.markdown("---")

D = st.sidebar.number_input("거리 (mm)", value=10000, step=500)
gain = st.sidebar.slider("밝기 Gain", 0.5, 3.0, 1.0, step=0.1)
manual_offset = st.sidebar.number_input("수동 Offset (mm)", value=0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("📱 촬영 기기 사양")
st.sidebar.info("기기: iPhone 15 Pro Max")

zoom_labels = list(IPHONE_15_PRO_MAX_SPECS.keys())
zoom_sel = st.sidebar.selectbox("촬영 배율 선택", zoom_labels)
FOV = IPHONE_15_PRO_MAX_SPECS[zoom_sel]

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
    
    # 세션 상태 고정 초기화
    if "roi_x1" not in st.session_state:
        st.session_state.roi_x1 = int(W_img * 0.35)
        st.session_state.roi_y1 = int(H_img * 0.35)
        st.session_state.roi_x2 = int(W_img * 0.65)
        st.session_state.roi_y2 = int(H_img * 0.65)
    
    # 🚨 [핵심 변경] 토글 상태에 따른 렌더링 분기
    if roi_edit_mode:
        st.info("🎯 [ROI 수정 모드] 이미지 위의 노란 상자를 움직여 분석 영역을 변경하세요.")
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
    else:
        # 🚨 [수정 모드가 껐을 때] 일반 정적 이미지를 노출시켜 터치 이벤트를 완전히 무력화(순정 스크롤 보장)
        st.success("📱 [정식 스크롤 모드] 화면을 자유롭게 위아래로 내리실 수 있습니다. 영역 수정을 원하시면 사이드바의 스위치를 켜세요.")
        
        # 현재 지정된 ROI를 정적 이미지 위에 미리보기용으로 가이드라인 매핑
        preview_img = np.array(origin_pil).copy()
        cv2.rectangle(
            preview_img, 
            (st.session_state.roi_x1, st.session_state.roi_y1), 
            (st.session_state.roi_x2, st.session_state.roi_y2), 
            (255, 221, 0), 6
        )
        st.image(preview_img, use_container_width=True)

    x1, y1, x2, y2 = st.session_state.roi_x1, st.session_state.roi_y1, st.session_state.roi_x2, st.session_state.roi_y2

    # 알고리즘: 원본 이미지 기반 vivid 컬러 마스킹 및 분석
    img_b = cv2.cvtColor(np.array(origin_pil), cv2.COLOR_RGB2BGR)
    orig_hsv = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    
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

        safe_margin = 30 
        roi_b = img_b[y1:y2, x1:x2]
        roi_g = np.clip(roi_b.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(cv2.GaussianBlur(gray, (9,9), 0), cv2.CV_64F, 0, 1, ksize=5)
        rel_c_y = int(c_y_auto - y1)
        grad[max(0, rel_c_y - safe_margin):min(roi_b.shape[0], rel_c_y + safe_margin), :] = 0
        l_y = np.argmax(np.mean(grad, axis=1)) + y1
        
        std_y = c_y + (math.degrees(math.atan(0.01)) / deg_per_px)
        mm_raw = D * math.tan(math.radians((c_y - l_y) * deg_per_px))
        mm_std = D * math.tan(math.radians((std_y - l_y) * deg_per_px))
        pct_raw, pct_std = (mm_raw / D) * 100, (mm_std / D) * 100
        
        st.markdown("---")
        view_col, graph_col = st.columns([1, 1])
        
        with view_col:
            st.subheader("🖼️ 분석 결과 이미지")
            disp_img = img_b.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 3.0 
            f_thick = 7

            cv2.line(disp_img, (0, int(c_y)), (W_img, int(c_y)), (0, 0, 255), 5)
            cv2.putText(disp_img, "LASER LINE", (50, int(c_y) - 30), font, f_scale, (0, 0, 255), f_thick)

            cv2.line(disp_img, (0, int(l_y)), (W_img, int(l_y)), (0, 255, 0), 5)
            cv2.putText(disp_img, "CUT-OFF", (50, int(l_y) - 30), font, f_scale, (0, 255, 0), f_thick)

            cv2.line(disp_img, (0, int(std_y)), (W_img, int(std_y)), (255, 0, 0), 4)
            cv2.putText(disp_img, "EU -1% Line", (W_img - 850, int(std_y) + 100), font, f_scale, (255, 0, 0), f_thick)

            cv2.rectangle(disp_img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            
            disp_img_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            st.image(disp_img_rgb, use_container_width=True)
            
        with graph_col:
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
            
        st.markdown("---")
        
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

        # 이미지 및 데이터 파일 인코딩 가공
        img_buffer = io.BytesIO()
        Image.fromarray(disp_img_rgb).save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        fig_buffer = io.BytesIO()
        fig.savefig(fig_buffer, format="PNG", bbox_inches='tight', facecolor=fig.get_facecolor())
        fig_bytes = fig_buffer.getvalue()

        report_text = (
            f"[Headlamp Cut-off Analyzer 분석 결과 레포트]\n\n"
            f"🇺🇸 북미 사양 결과:\n"
            f" - 오차값: {mm_raw:+.1f} mm ({pct_raw:+.2f}%)\n"
            f" - 최종 판정: {us_status}\n\n"
            f"🇪🇺 유럽 사양 결과:\n"
            f" - 오차값: {mm_std:+.1f} mm ({pct_std:+.2f}%)\n"
            f" - 최종 판정: {eu_status}\n"
        )
        report_bytes = report_text.encode('utf-8')

        # 사이드바 다운로드 트리거 패널
        st.sidebar.markdown("---")
        st.sidebar.subheader("📸 결과 데이터 다운로드")
        
        ios_safe_download_button("🖼️ 분석 완료 이미지 받기", img_bytes, "headlamp_analysis.png", "image/png")
        ios_safe_download_button("📊 그래프 프로필 받기", fig_bytes, "intensity_profile.png", "image/png")
        ios_safe_download_button("📋 판정 결과 레포트 받기", report_bytes, "judgment_report.txt", "text/plain")

    else:
        st.sidebar.markdown("---")
        st.sidebar.warning("⚠️ 이미지가 올바르게 분석되지 않아 다운로드 버튼을 활성화할 수 없습니다.")
        st.error("레이저 라인을 인식하지 못했습니다. vivid 컬러의 라인을 ROI 내에 포함시키거나, 밝기 Gain을 조절해 주세요.")
else:
    st.sidebar.markdown("---")
    st.sidebar.info("💡 메인 화면에 이미지를 업로드하시면 결과 저장 버튼이 활성화됩니다.")
