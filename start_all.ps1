# RadPilot Agent Workstation PowerShell 原生启动器 (避免 Windows Terminal cmd 0x800700e8 报错)
$Host.UI.RawUI.WindowTitle = "RadPilot Launcher"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "        RadPilot Agent Workstation Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. 正在启动 Python 后端服务 (Port 8000)..." -ForegroundColor Yellow
Start-Process "C:\Users\Ostiu\AppData\Local\Programs\Python\Python312\python.exe" -ArgumentList "backend/main.py" -WorkingDirectory "d:\Project\RadPilot"

Write-Host "2. 正在启动 Vite 前端服务 (Port 5173)..." -ForegroundColor Yellow
Start-Process "powershell" -ArgumentList "-NoExit -Command cd d:\Project\RadPilot\frontend; npm run dev" -WorkingDirectory "d:\Project\RadPilot\frontend"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host " 服务启动完成！请在浏览器访问: http://localhost:5173" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
