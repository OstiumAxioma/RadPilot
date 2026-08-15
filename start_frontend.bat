@echo off
chcp 65001 >nul
title RadPilot Frontend
echo Starting React Vite Frontend on Port 5173...
cd /d "%~dp0frontend"
npm run dev
pause
