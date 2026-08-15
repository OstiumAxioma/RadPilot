@echo off
chcp 65001 >nul
title RadPilot Process Cleaner

echo Cleaning RadPilot processes on Port 8000 & 5173...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Killing backend PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do (
    echo Killing frontend PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

echo Cleanup completed!
