@echo off
rem 无黑框启动薄荷（猫猫）桌宠
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
start "" pyw -3 -m mint_pet
