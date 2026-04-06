@echo off
REM AetherArch Local CI Test Runner for Windows
REM Runs all GitHub Actions CI jobs in Docker containers

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Get git info
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD') do set BRANCH=%%i
for /f "tokens=*" %%i in ('git rev-parse --short HEAD') do set COMMIT=%%i

set IMAGE_NAME=aether-ci-test
set CONTAINER_NAME=aether-ci-%BRANCH%-%COMMIT%
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)

echo.
echo ================================================================
echo           AetherArch Local CI Test Runner
echo ================================================================
echo.
echo Branch:          %BRANCH%
echo Commit:          %COMMIT%
echo Container Name:  %CONTAINER_NAME%
echo Timestamp:       %mydate%-%mytime%
echo.

REM Check Docker availability
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop for Windows
    exit /b 1
)

echo Building Docker image...
docker build -f Dockerfile.ci -t %IMAGE_NAME% . || (
    echo ERROR: Docker build failed
    exit /b 1
)
echo.
echo Docker image built successfully: %IMAGE_NAME%
echo.

REM Run all tests
echo ================================================================
echo Running all CI tests...
echo ================================================================
echo.

docker run ^
    --rm ^
    --name %CONTAINER_NAME% ^
    --cpus 4 ^
    --memory 8g ^
    -v "%cd%:/workspace" ^
    %IMAGE_NAME%

if errorlevel 1 (
    echo.
    echo ERROR: CI tests failed!
    exit /b 1
)

echo.
echo ================================================================
echo ALL CI TESTS PASSED SUCCESSFULLY!
echo ================================================================
echo.
echo Test Results:
echo   [OK] Build test
echo   [OK] Test suite
echo   [OK] Clippy linting
echo   [OK] Format check
echo   [OK] Documentation
echo   [OK] MSRV check
echo.
echo Next Steps:
echo   1. Push to feature branch: git push origin %BRANCH%
echo   2. Create pull request to main
echo   3. GitHub Actions will run additional platform tests
echo.

endlocal
