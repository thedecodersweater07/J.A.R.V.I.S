@echo off
cd server\web
call npm install
call npm run build
cd ..\..
