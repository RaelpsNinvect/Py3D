import os
import torch
import numpy as np
import trimesh
from PIL import Image
from torchvision import transforms
from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageUploadForm

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
    # Resize the image to a more manageable size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjusted to a different size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Estimate depth from the image tensor
def estimate_depth(model, image_tensor):
    print("Estimating depth...")
    print("Input tensor shape:", image_tensor.shape)  # Debug input tensor shape
    with torch.no_grad():
        depth = model(image_tensor)
    print("Depth output shape:", depth.shape)  # Debug output depth shape

    depth = depth.squeeze().cpu().numpy()

    # Ensure depth map has the expected dimensions
    print("Depth map shape before resizing:", depth.shape)
    depth = np.clip(depth, 0, np.max(depth))  # Clip to remove negative values
    depth = depth / np.max(depth)  # Normalize between 0 and 1

    print("Depth map shape after normalization:", depth.shape)
    return depth

# Create a 3D model from the depth map
def create_3d_model(depth_map):
    print("Creating 3D model...")
    h, w = depth_map.shape
    x, y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    
    # Adjust the z-coordinate for proper orientation
    z = depth_map * 1000  # Scale to a suitable range

    # Create vertices without scaling by aspect ratio
    vertices = np.column_stack((
        x.flatten(),  # No scaling applied
        -y.flatten(),  # Negate y for proper orientation
        z.flatten()
    ))  
    
    # Define faces (using grid connectivity)
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            # Define two triangles for each square
            bottom_left = i * w + j
            bottom_right = bottom_left + 1
            top_left = (i + 1) * w + j
            top_right = top_left + 1
            
            # Append faces in counter-clockwise order for proper normals
            faces.append([bottom_left, top_left, bottom_right])  # Lower triangle
            faces.append([bottom_right, top_left, top_right])    # Upper triangle

    faces = np.array(faces)

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print("Mesh created with", len(vertices), "vertices and", len(faces), "faces.")
    
    return mesh

# Save the 3D model to a file
def save_3d_model(mesh, output_path):
    print("Saving 3D model to:", output_path)
    mesh.export(output_path)

# Convert the uploaded image to a 3D model
def convert_to_3d(image_path, output_path):
    model = load_midas_model()
    if model is None:
        print("Model loading failed.")
        return
    
    image_tensor = preprocess_image(image_path)
    depth_map = estimate_depth(model, image_tensor)
    
    if depth_map is None or depth_map.size == 0:
        print("Depth map is empty or None.")
        return
    
    print("Depth map shape:", depth_map.shape)
    mesh = create_3d_model(depth_map)
    
    if mesh is None:
        print("Mesh creation failed.")
        return
    
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
            image_path = os.path.join('media/uploads', image.name)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Save the uploaded file
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            output_path = image_path.replace('.jpg', '.obj').replace('.png', '.obj')
            print("Starting conversion for:", image_path)
            convert_to_3d(image_path, output_path)

            # Check if the output model file exists
            if os.path.exists(output_path):
                response = HttpResponse(open(output_path, 'rb').read())
                response['Content-Disposition'] = f'attachment; filename="{os.path.basename(output_path)}"'
                response['Content-Type'] = 'application/octet-stream'
                return response
            else:
                return HttpResponse("3D model could not be created.", status=500)
        else:
            return HttpResponse(f'Failed to upload file. Errors: {form.errors}', status=400)
    return HttpResponse('No file uploaded.', status=400)

# Generate a 3D mesh from an image
def generate_mesh(request):
    if request.method == 'POST':
        image_path = request.POST.get('image_path')
        
        # Validate the image path
        if not image_path or not os.path.isfile(image_path):
            return HttpResponse("Invalid image path.", status=400)

        output_path = image_path.replace('.jpg', '.obj').replace('.png', '.obj')

        # Call your conversion function
        try:
            convert_to_3d(image_path, output_path)
            if os.path.exists(output_path):
                return HttpResponse(f"Mesh generated successfully! You can find the 3D model in '{output_path}'.")
            else:
                return HttpResponse("3D model could not be created.", status=500)
        except Exception as e:
            return HttpResponse(f"An error occurred during mesh generation: {str(e)}", status=500)

    return HttpResponse("Invalid request method.", status=400)
