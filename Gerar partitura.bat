@echo off
chcp 65001 >nul
if "%~1"=="" (
    echo Arraste uma musica ou video em cima deste arquivo.
    pause
    exit /b
)
echo Musica: %~nx1
echo.
set VEL=1
set /p VEL="Velocidade (Enter = normal, 0.75 = mais lenta, 0.5 = bem lenta): "
set CLIQUE=
set /p RESP="Colocar som de TUM/TA por cima da musica? (s/N): "
if /i "%RESP%"=="s" set CLIQUE=--clique
set AULA=
set RESP=
set /p RESP="Gerar as aulas passo a passo de cada levada e os cortes dos trechos? (s/N): "
if /i "%RESP%"=="s" set AULA=--trechos
set TRES=
set RESP=
set /p RESP="Separar os toques de borda em TOM (aberto) e TA (tapa)? So pra quem ja faz os 3 sons (s/N): "
if /i "%RESP%"=="s" set TRES=--tres-sons
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0partitura.py" "%~1" --velocidade %VEL% %CLIQUE% %AULA% %TRES%
echo.
start "" "%~dp0saida"
pause
