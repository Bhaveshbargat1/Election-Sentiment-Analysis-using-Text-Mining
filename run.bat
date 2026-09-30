@echo off
title Election Social Media Sentiment Analysis - Web Server
echo ======================================================================
echo  VotePulse AI: Election Sentiment Analysis using Text Mining
echo ======================================================================
echo Starting local web server...
python run.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Trying with Python 3.11...
    py -3.11 run.py
)
pause
