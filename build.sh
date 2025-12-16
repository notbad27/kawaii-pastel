#!/bin/bash

# Docker Build and Push Script for Kawaii Pastel Worker
# Make sure Docker is running before executing this script

DOCKER_USERNAME="notbad27"
IMAGE_NAME="kawaii-pastel-worker"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}"

echo "========================================"
echo "Building Kawaii Pastel Worker Docker Image"
echo "========================================"
echo ""

# Check if Docker is available
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "✗ Docker not found! Please install Docker."
    exit 1
fi

echo "✓ Docker found: $(docker --version)"
echo ""
echo "Building Docker image: $FULL_IMAGE_NAME"
echo "This may take several minutes..."
echo ""

# Build the Docker image
docker build --platform linux/amd64 --tag $FULL_IMAGE_NAME:latest .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "Pushing image to Docker Hub..."
    echo "Make sure you're logged in: docker login"
    echo ""
    
    # Push the image
    docker push $FULL_IMAGE_NAME:latest
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "✓ SUCCESS! Image pushed to Docker Hub"
        echo "========================================"
        echo ""
        echo "Image URL: docker.io/$FULL_IMAGE_NAME:latest"
        echo ""
        echo "Next steps:"
        echo "1. Go to RunPod Console: https://www.console.runpod.io/serverless"
        echo "2. Click 'New Endpoint' or edit existing endpoint"
        echo "3. Use this image: docker.io/$FULL_IMAGE_NAME:latest"
        echo ""
    else
        echo ""
        echo "✗ Push failed. Make sure you're logged in:"
        echo "  docker login"
    fi
else
    echo ""
    echo "✗ Build failed! Check the error messages above."
    exit 1
fi

