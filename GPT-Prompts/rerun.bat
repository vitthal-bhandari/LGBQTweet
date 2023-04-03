@echo off
:loop
start /wait cmd.exe /c python main_script.py > error_log.txt 2>&1
timeout /t 60
goto loop