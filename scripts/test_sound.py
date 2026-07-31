"""Quick test: QSoundEffect playback"""
import sys, os, time
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, r'C:\Users\HP\Desktop\drum项目\drum-ring')

from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)
sound_dir = Path(r'C:\Users\HP\Desktop\drum项目\drum-ring\assets\drum_gesture\sounds')

for wav_name in sorted(sound_dir.glob('*.wav')):
    if wav_name.suffix == '.bak':
        continue
    effect = QSoundEffect()
    url = QUrl.fromLocalFile(str(wav_name.resolve()))
    effect.setSource(url)
    effect.setVolume(0.8)
    print(f'{wav_name.name}: ', end='', flush=True)
    
    for _ in range(200):
        app.processEvents()
        time.sleep(0.01)
        if effect.status() != QSoundEffect.Status.Loading:
            break
    
    status_names = {0: 'Null', 1: 'Loading', 2: 'Ready', 3: 'Error'}
    status = effect.status()
    print(f'{status_names.get(status, str(status))}', flush=True)
    
    if status == QSoundEffect.Status.Ready:
        effect.play()
        print(f'  -> play() called', flush=True)
        time.sleep(0.3)
        app.processEvents()

print('\nDone.', flush=True)
