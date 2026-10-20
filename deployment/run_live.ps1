# Starts the local live backend for the website's arena (GPU required).
# Run from the project root:  .\deployment\run_live.ps1
# Then open the website's Workbench page; it connects to this server and lets you ask the 4 bit models live. Stop with Ctrl+C.
& "$PSScriptRoot\..\venv\Scripts\python.exe" -m uvicorn deployment.server:app --host 127.0.0.1 --port 8000
