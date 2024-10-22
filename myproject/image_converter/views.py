import os
import torch
import numpy as np
import trimesh
from PIL import Image
from torchvision import transforms
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .forms import ImageUploadForm
from django.conf import settings

# Load the MiDaS model
def load_midas_model():
    print("Loading MiDaS model...")
    model = torch.hub.load("isl-org/MiDaS", "MiDaS_small")  # Using a specific version
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path):
    print("Preprocessing image...")
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjusted to a different size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Estimate depth from the image tensor
def estimate_depth(model, image_tensor):
    print("Estimating depth...")
    with torch.no_grad():
        depth = model(image_tensor)
    depth = depth.squeeze().cpu().numpy()
    depth = np.clip(depth, 0, np.max(depth))  # Clip to remove negative values
    depth = depth / np.max(depth)  # Normalize between 0 and 1
    return depth

# Create a 3D model from the depth map
def create_3d_model(depth_map):
    print("Creating 3D model...")
    h, w = depth_map.shape
    x, y = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    z = depth_map * 300  # Scale to a suitable range

    vertices = np.column_stack((
        x.flatten(),
        -y.flatten(),
        z.flatten()
    ))
    
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            bottom_left = i * w + j
            bottom_right = bottom_left + 1
            top_left = (i + 1) * w + j
            top_right = top_left + 1
            faces.append([bottom_left, top_left, bottom_right])  # Lower triangle
            faces.append([bottom_right, top_left, top_right])    # Upper triangle

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

# Save the 3D model to a file
def save_3d_model(mesh, output_path):
    print("Saving 3D model to:", output_path)
    mesh.export(output_path)

# Convert the uploaded image to a 3D model
def convert_to_3d(image_path, output_path):
    model = load_midas_model()
    image_tensor = preprocess_image(image_path)
    depth_map = estimate_depth(model, image_tensor)
    mesh = create_3d_model(depth_map)
    save_3d_model(mesh, output_path)

# Render the index page
def index(request):
    form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})

# Handle image upload and conversion
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image.name)

            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            output_path = image_path.replace('.jpg', '.obj').replace('.png', '.obj')
            convert_to_3d(image_path, output_path)

            if os.path.exists(output_path):
                return JsonResponse({'obj_url': f'/media/uploads/{os.path.basename(output_path)}'})
            else:
                return JsonResponse({'error': '3D model could not be created.'}, status=500)
        else:
            return JsonResponse({'error': f'Failed to upload file. Errors: {form.errors}'}, status=400)
    return JsonResponse({'error': 'No file uploaded.'}, status=400)
