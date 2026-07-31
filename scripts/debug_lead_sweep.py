"""Debug: show per-frame probability sweep around one peak."""
import sys, json, base64, tempfile, numpy as np, torch
from pathlib import Path
sys.path.insert(0, r'C:\Users\HP\Desktop\drum项目\drum-ring')
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from models.drum_gesture_recognizer import DrumGestureRecognizer, OnlineLinearAccelerationEstimator, DRUM_CLASS_LABELS
import pandas as pd

ROOT = Path(r'C:\Users\HP\Desktop\drum项目\drum-ring')
_INDEX_PATH = ROOT / 'apps' / 'shared' / '.ui_index.json'
idx = json.loads(_INDEX_PATH.read_text(encoding='utf-8'))
phrase = ''.join(chr(a ^ b) for a, b in zip(idx['left'], idx['right']))
stamp_raw = base64.b64decode(idx['stamp'])

data_dir = ROOT / 'data'
tmp = tempfile.TemporaryDirectory(prefix='drum_debug_')
dec_dir = Path(tmp.name)
print("Decrypting...", flush=True)

for path in sorted(data_dir.glob('*.enc')):
    if '1783343401' not in path.name:
        continue
    blob = path.read_bytes()
    candidates = [stamp_raw]
    if stamp_raw.endswith(b'\n'):
        candidates.append(stamp_raw[:-1] + b'\r\n')
    for st in candidates:
        if blob.startswith(st):
            payload_bytes = blob[len(st):]
            break
    payload = json.loads(payload_bytes.decode('utf-8'))
    salt = base64.b64decode(payload['salt'])
    token = str(payload['blob']).encode('ascii')
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=int(payload['rounds']))
    key = base64.urlsafe_b64encode(kdf.derive(phrase.encode('utf-8')))
    data = Fernet(key).decrypt(token)
    (dec_dir / Path(str(payload['name']).replace(':', '_')).name).write_bytes(data)

print("Loading...", flush=True)
imu_path = list(dec_dir.glob('*zhy_1.npy'))[0]
serial_path = list(dec_dir.glob('serial*zhy_1.csv'))[0]
raw_imu = np.load(imu_path)
if raw_imu.shape[1] > 1 and np.nanmax(raw_imu[:, 1]) > 0:
    t_rel = raw_imu[:, 1].astype(np.float64) - float(raw_imu[0, 1])
else:
    t_rel = raw_imu[:, 0].astype(np.float64) - float(raw_imu[0, 0])

df = pd.read_csv(serial_path)
mask = df['line'].astype(str).str.contains('FALLING', na=False)
events = df.loc[mask, 'elapsed_s'].to_numpy(dtype=np.float64)

print("Estimating linear acc...", flush=True)
acc = raw_imu[:, 2:5].astype(np.float32)
gyro = raw_imu[:, 5:8].astype(np.float32)
estimator = OnlineLinearAccelerationEstimator()
linear_imu = np.zeros((raw_imu.shape[0], 6), dtype=np.float32)
last_t = float(t_rel[0])
for idx in range(raw_imu.shape[0]):
    dt = float(t_rel[idx]) - last_t if idx else 1.0 / 200.0
    la, _ = estimator.update(acc[idx], gyro[idx], dt)
    linear_imu[idx, :3] = la
    linear_imu[idx, 3:] = gyro[idx]
    last_t = float(t_rel[idx])

print("Aligning peaks...", flush=True)
PEAK_SEARCH = 30
peaks = []
for et in events:
    ei = int(np.argmin(np.abs(t_rel - et)))
    s, e = ei - PEAK_SEARCH, ei + PEAK_SEARCH + 1
    if s >= 0 and e <= linear_imu.shape[0]:
        peaks.append(s + int(np.argmax(np.linalg.norm(linear_imu[s:e, :3], axis=1))))

print("Loading model...", flush=True)
recognizer = DrumGestureRecognizer(
    str(ROOT / 'checkpoint' / 'drum_gesture' / 'local_drum_all_w15_lead22201816_rawlin12hc_weighted_balanced_random.pth'),
    device='cpu')
WINDOW = 15
LEADS = (22, 20, 18, 16)
model = recognizer.model
model.eval()

print("\n" + "=" * 110, flush=True)
print("Per-peak PROBABILITY CURVE (3 peaks: #3, #5, #7)", flush=True)
print("=" * 110, flush=True)

for peak_num in [2, 4, 6]:  # 0-indexed peaks 3, 5, 7
    peak_idx = peaks[peak_num]
    print(f"\n--- Peak #{peak_num + 1}: frame={peak_idx}, time={t_rel[peak_idx]:.3f}s ---", flush=True)
    print(f"Training leads: end at frames {[peak_idx - l for l in LEADS]}", flush=True)
    print(f"Lead range: [{peak_idx - 22}, {peak_idx - 16}] (lead 22 to 16)", flush=True)
    print(f"{'end_fr':>7} {'pred':>7} {'conf':>6} | {'bg(0)':>7} {'踩镲(1)':>7} {'通2(2)':>7} {'军(3)':>7} {'通1(4)':>7} | NOTE", flush=True)
    print("-" * 105, flush=True)

    windows = []
    end_frames = []
    for end_inclusive in range(peak_idx - 35, peak_idx - 5):
        start = end_inclusive - WINDOW + 1
        raw6 = raw_imu[start:end_inclusive + 1, 2:8].astype(np.float32)
        lin6 = linear_imu[start:end_inclusive + 1, :].astype(np.float32)
        win = np.concatenate([raw6, lin6], axis=1)
        windows.append(recognizer.prepare(win))
        end_frames.append(end_inclusive)

    prepared = np.stack(windows, axis=0)
    inputs = torch.from_numpy(prepared)
    with torch.inference_mode():
        probs = torch.softmax(model(inputs), dim=1).cpu().numpy()

    first_detect_frame = None
    max_conf_frame = None
    max_conf_val = 0
    for ef, pr in zip(end_frames, probs):
        cls = int(np.argmax(pr))
        conf = float(pr[cls])
        marker = ""
        if cls == 1 and conf >= 0.8:
            if first_detect_frame is None:
                first_detect_frame = ef
            if conf > max_conf_val:
                max_conf_val = conf
                max_conf_frame = ef
            marker = " *** DETECT" if conf >= 0.9 else " ** CONFIDENT"
        if ef in [peak_idx - l for l in LEADS]:
            marker += f" [LEAD={peak_idx - ef}]"
        print(f"{ef:>7} {DRUM_CLASS_LABELS.get(cls, '?'):>7} {conf:>6.3f} | {pr[0]:>7.3f} {pr[1]:>7.3f} {pr[2]:>7.3f} {pr[3]:>7.3f} {pr[4]:>7.3f} {marker}", flush=True)

    print(f"\nFirst confident detection (>=0.8): frame {first_detect_frame} (lead={peak_idx - first_detect_frame if first_detect_frame else 'N/A'})", flush=True)
    print(f"Max confidence for class 1: {max_conf_val:.3f} at frame {max_conf_frame} (lead={peak_idx - max_conf_frame if max_conf_frame else 'N/A'})", flush=True)

tmp.cleanup()
print("\nDone.", flush=True)
