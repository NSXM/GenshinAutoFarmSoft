@echo off
REM ============================================================================
REM  Установка окружения. Python 3.12 (НЕ 3.14 — под него ещё нет готовых колёс
REM  numpy/torch, pip полезет компилировать и упадёт без компилятора).
REM  Диск C: переполнен -> весь pip temp/cache гоним на G:.
REM
REM  ВАЖНО: путь к Python берём из %LOCALAPPDATA% (уже разрешённое окружением
REM  значение), а НЕ через C:\Users\%USERNAME%\... — имя пользователя кириллицей
REM  ломало подстановку и venv молча откатывался на системный 3.14.
REM ============================================================================
setlocal enabledelayedexpansion

set PROJ=G:\genshin_auto_nav

REM --- ищем Python 3.12 ---
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY312%" (
    REM запасной вариант — лаунчер
    py -3.12 -c "import sys" 1>nul 2>nul && set "PY312=py -3.12"
)
if not exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" if not "%PY312%"=="py -3.12" (
    echo [setup] ОШИБКА: Python 3.12 не найден. Установи его и повтори.
    exit /b 1
)

REM --- pip temp/cache на G: ---
set TMP=G:\Temp
set TEMP=G:\Temp
set PIP_CACHE_DIR=G:\pipcache
if not exist "G:\Temp" mkdir "G:\Temp"
if not exist "G:\pipcache" mkdir "G:\pipcache"

cd /d %PROJ%

if exist ".venv\Scripts\python.exe" (
    echo [setup] .venv уже существует — пропускаю создание.
) else (
    echo [setup] создаю venv на Python 3.12...
    %PY312% -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo [setup] ОШИБКА: venv не создался. Прерываю — НЕ откатываюсь на 3.14.
        exit /b 1
    )
)

REM --- работаем напрямую через python из venv, без activate ---
set "VENV_PY=%PROJ%\.venv\Scripts\python.exe"

REM --- жёсткая проверка: это точно 3.12, а не системный 3.14 ---
"%VENV_PY%" -c "import sys; assert sys.version_info[:2]==(3,12), sys.version; print('[setup] venv Python', sys.version.split()[0])"
if errorlevel 1 (
    echo [setup] ОШИБКА: в venv не Python 3.12. Прерываю.
    exit /b 1
)

"%VENV_PY%" -m pip install --upgrade pip --cache-dir G:\pipcache

echo [setup] базовые зависимости...
"%VENV_PY%" -m pip install --cache-dir G:\pipcache -r requirements.txt
if errorlevel 1 (
    echo [setup] ОШИБКА установки зависимостей. Смотри лог выше.
    exit /b 1
)

echo.
echo [setup] Базовое окружение готово. Отдельно (под своё железо) установить:
echo   torch+CUDA:   "%VENV_PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --cache-dir G:\pipcache
echo   transformers: "%VENV_PY%" -m pip install transformers --cache-dir G:\pipcache
echo   SAM3 веса:    "%VENV_PY%" -m huggingface_hub login  затем  huggingface-cli download facebook/sam3
echo   OpenVINO:     "%VENV_PY%" -m pip install openvino --cache-dir G:\pipcache
echo.
echo [setup] Запуск:
echo   .venv\Scripts\activate
echo   python run.py --config config.yaml --route routes\example_route.json
endlocal
