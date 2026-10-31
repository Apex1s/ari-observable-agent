@echo off
rem Double-click to talk to Ari (English). Add --text to type instead of speaking.
cd /d "%~dp0"
python -c "import sounddevice, edge_tts, speech_recognition, pyttsx3, fastapi, numpy" 2>nul || python -m pip install -q -r requirements-talk.txt
python -m ari.talk %*
pause
