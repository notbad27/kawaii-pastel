# Docker Build and Push Script for Kawaii Pastel Worker
# Make sure Docker Desktop is running before executing this script

$DOCKER_USERNAME = "notbad27"
$IMAGE_NAME = "kawaii-pastel-worker"
$FULL_IMAGE_NAME = "${DOCKER_USERNAME}/${IMAGE_NAME}"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Kawaii Pastel Worker Docker Image" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is available
Write-Host "Checking Docker installation..." -ForegroundColor Yellow
$dockerCheck = docker --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker found: $dockerCheck" -ForegroundColor Green
} else {
    Write-Host "✗ Docker not found! Please install Docker Desktop and make sure it's running." -ForegroundColor Red
    Write-Host "Error: $dockerCheck" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Building Docker image: $FULL_IMAGE_NAME" -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow
Write-Host ""

# Build the Docker image
docker build --platform linux/amd64 --tag $FULL_IMAGE_NAME:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Pushing image to Docker Hub..." -ForegroundColor Yellow
    Write-Host "Make sure you're logged in: docker login" -ForegroundColor Yellow
    Write-Host ""
    
    # Push the image
    docker push $FULL_IMAGE_NAME:latest
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✓ SUCCESS! Image pushed to Docker Hub" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Image URL: docker.io/$FULL_IMAGE_NAME:latest" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Go to RunPod Console: https://www.console.runpod.io/serverless" -ForegroundColor White
        Write-Host "2. Click 'New Endpoint' or edit existing endpoint" -ForegroundColor White
        Write-Host "3. Use this image: docker.io/$FULL_IMAGE_NAME:latest" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "✗ Push failed. Make sure you're logged in:" -ForegroundColor Red
        Write-Host "  docker login" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "✗ Build failed! Check the error messages above." -ForegroundColor Red
    exit 1
}
