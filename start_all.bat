@echo off
chcp 65001 >nul
echo Starting RadPilot Workstation...
powershell -ExecutionPolicy Bypass -File "d:\Project\RadPilot\start_all.ps1"
