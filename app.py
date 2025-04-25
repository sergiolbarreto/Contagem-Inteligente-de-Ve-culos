# app.py

import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.spatial import distance as dist

# ——————————————
# Thresholds fixos para detecção do semáforo (HSV)
GREEN_LOWER = np.array([40, 40, 40])
GREEN_UPPER = np.array([80, 255, 255])
RED_LOWER1  = np.array([0, 50, 50])
RED_UPPER1  = np.array([10, 255, 255])
RED_LOWER2  = np.array([170, 50, 50])
RED_UPPER2  = np.array([180, 255, 255])
# ——————————————

def preview_frame(frame, line_ratio, start_ratio, end_ratio):
    """ Desenha a linha virtual para preview. """
    h, w = frame.shape[:2]
    y = int(h * line_ratio)
    x0 = int(w * start_ratio)
    x1 = int(w * end_ratio)
    cv2.line(frame, (x0, y), (x1, y), (0, 0, 255), 2)
    return frame

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, rects):
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        centroids = np.array([[(x1+x2)//2, (y1+y2)//2] for x1,y1,x2,y2 in rects])
        if not self.objects:
            for c in centroids:
                self.register(tuple(c))
        else:
            ids = list(self.objects)
            prev = list(self.objects.values())
            D = dist.cdist(np.array(prev), centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()
            for r, c in zip(rows, cols):
                if r in used_r or c in used_c or D[r, c] > self.max_distance:
                    continue
                oid = ids[r]
                self.objects[oid] = tuple(centroids[c])
                self.disappeared[oid] = 0
                used_r.add(r); used_c.add(c)
            for r in set(range(D.shape[0])) - used_r:
                oid = ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            for c in set(range(D.shape[1])) - used_c:
                self.register(tuple(centroids[c]))
        return self.objects

def detect_traffic_light_state(
    frame, tl_model,
    default='VERMELHO'
):
    """
    Detecta se o semáforo está VERDE ou VERMELHO usando thresholds fixos.
    """
    results = tl_model.predict(source=frame, imgsz=640, conf=0.5)
    boxes   = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confs   = results[0].boxes.conf.cpu().numpy()
    idxs    = np.where(classes == 9)[0]  # classe 9 = traffic light
    if len(idxs) == 0:
        return default

    best = idxs[np.argmax(confs[idxs])]
    x1,y1,x2,y2 = boxes[best].astype(int)
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    g = cv2.countNonZero(cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER))
    r1 = cv2.countNonZero(cv2.inRange(hsv, RED_LOWER1, RED_UPPER1))
    r2 = cv2.countNonZero(cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))

    return 'VERDE' if g > (r1 + r2) else 'VERMELHO'

def process_video(input_path, params):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Não foi possível abrir o vídeo.")
        return None, None, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_y   = int(h * params['line_ratio'])
    x0       = int(w * params['start_ratio'])
    x1       = int(w * params['end_ratio'])
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_vid = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out_vid = cv2.VideoWriter(tmp_vid.name, fourcc, fps, (w, h))

    tl_model = YOLO(params['model_path'])
    vm       = YOLO(params['model_path'])
    tracker  = CentroidTracker()
    memory, counted, records = {}, set(), []
    last_state = 'VERMELHO'
    frame_id = 0
    progress = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Estado do semáforo
        state = detect_traffic_light_state(frame, tl_model, default=last_state)
        last_state = state

        res = vm.predict(source=frame, imgsz=640, conf=0.5)
        bxs = res[0].boxes.xyxy.cpu().numpy()
        cls = res[0].boxes.cls.cpu().numpy().astype(int)
        mask = np.isin(cls, [2,3,5,7])
        rects = [tuple(map(int, b)) for b in bxs[mask]]

        objs = tracker.update(rects)
        cv2.line(frame, (x0, line_y), (x1, line_y), (255,0,0), 2)

        for oid, (cx, cy) in objs.items():
            prev = memory.get(oid, cy)
            if (oid not in counted
                and ((prev < line_y <= cy) or (prev > line_y >= cy))
                and state == 'VERDE'):
                counted.add(oid)
                records.append({
                    'id': oid,
                    'timestamp_s': frame_id / fps,
                    'frame': frame_id,
                    'state': state
                })
            memory[oid] = cy
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

        cv2.putText(frame, f"{state} | Cont: {len(counted)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
        out_vid.write(frame)

        frame_id += 1
        progress.progress(min(frame_id/total_f, 1.0))

    cap.release()
    out_vid.release()

    df = pd.DataFrame(records)
    tmp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    df.to_csv(tmp_csv.name, index=False)
    return tmp_vid.name, tmp_csv.name, len(df)

# ——————————————
# Streamlit App
# ——————————————

st.title("🚦 Contagem Inteligente de Veículos")

# Upload do vídeo
st.sidebar.header("Upload do Vídeo")
VIDEO_FILE = st.sidebar.file_uploader("Vídeo (mp4)", type=["mp4"])

model_path = "yolov8n.pt"

# Parâmetros da linha virtual
st.sidebar.header("Linha Virtual")
line_ratio  = st.sidebar.slider("Altura da linha (Y)",  0.0, 1.0, 0.75, key="line_ratio")
start_ratio = st.sidebar.slider("Início da linha (X%)", 0.0, 1.0, 0.10, key="start_ratio")
end_ratio   = st.sidebar.slider("Fim da linha (X%)",   0.0, 1.0, 0.80, key="end_ratio")

if VIDEO_FILE is not None:
    # salva upload em arquivo temporário
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(VIDEO_FILE.read())

    # preview da linha
    cap = cv2.VideoCapture(tfile.name)
    ret, frame = cap.read()
    cap.release()
    if ret:
        st.subheader("👁️ Preview da Linha Virtual")
        vis = preview_frame(frame.copy(), line_ratio, start_ratio, end_ratio)
        st.image(vis[:, :, ::-1], use_column_width=True)

    # botão de processamento
    if st.button("▶ Processar Vídeo"):
        params = {
            'model_path':  model_path,
            'line_ratio':  line_ratio,
            'start_ratio': start_ratio,
            'end_ratio':   end_ratio
        }
        if "video_out" not in st.session_state or "csv_out" not in st.session_state:
            video_out, csv_out, total = process_video(tfile.name, params)
            st.session_state.video_out = video_out
            st.session_state.csv_out = csv_out
            st.session_state.total = total

            # Também podemos guardar o DataFrame se quiser evitar reler o CSV
            st.session_state.df = pd.read_csv(csv_out)
            st.success(f"Processamento concluído! Total de veículos: {total}")
        else:
            st.success(f"Processamento concluído! Total de veículos: {st.session_state.total}")

        st.subheader("▶ Vídeo Processado")
        st.video(st.session_state.video_out)

        # Botão de download do vídeo
        st.download_button(
            "📥 Baixar Vídeo",
            data=open(st.session_state.video_out, "rb"),
            file_name="video_processado.mp4",
            mime="video/mp4"
        )

        st.subheader("📊 Dados de Contagem")
        st.dataframe(st.session_state.df)

        st.download_button(
            "📥 Baixar CSV",
            data=open(st.session_state.csv_out, "rb"),
            file_name="contagem_veiculos.csv",
            mime="text/csv"
        )
else:
    st.info("Faça upload de um vídeo mp4 na sidebar para começar.")
