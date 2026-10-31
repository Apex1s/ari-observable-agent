@echo off
rem Double-click: press Enter before each question, then speak. Most reliable in a noisy room.
cd /d "%~dp0"
python -c "import sounddevice, edge_tts, speech_recognition, pyttsx3, fastapi, numpy" 2>nul || python -m pip install -q -r requirements-talk.txt
python -m ari.talk --ptt %*
pause
