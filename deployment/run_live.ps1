# Starts the local live backend for the website's arena (GPU required).
# Run from the project root:  .\deployment\run_live.ps1
# Then open the website; its Live tab appears when this server is running. Stop with Ctrl+C.
& "$PSScriptRoot\..\venv\Scripts\python.exe" -m uvicorn deployment.server:app --host 127.0.0.1 --port 8000
