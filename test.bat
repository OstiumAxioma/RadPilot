@echo off
C:\Users\Ostiu\AppData\Local\Programs\Python\Python312\python.exe backend/test_backend.py > test_result.log 2>&1
echo DONE_TEST %ERRORLEVEL%
