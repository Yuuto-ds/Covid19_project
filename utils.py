import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2

# Load the trained CNN model
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.BatchNorm1d(512),
        nn.Linear(512, 3)  # Output layer for 3 classes
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# Generate Grad-CAM with TURBO colormap
def generate_gradcam(model, image_tensor, target_class, mask):
    gradients = None
    feature_maps = None

    # Hook to capture gradients
    def save_gradients(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Hook to capture feature maps
    def save_feature_maps(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    # Register hooks on the last convolutional layer
    target_layer = model.layer4[2].conv3
    forward_hook = target_layer.register_forward_hook(save_feature_maps)
    backward_hook = target_layer.register_backward_hook(save_gradients)

    # Forward pass
    output = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    # Remove hooks
    forward_hook.remove()
    backward_hook.remove()

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(feature_maps.shape[1]):
        feature_maps[0, i] *= pooled_gradients[i]

    heatmap = torch.mean(feature_maps[0], dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize heatmap to match the image size
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Apply lung mask
    mask = np.array(mask.resize((256, 256))) > 0
    heatmap[~mask] = 0

    # Convert the image tensor to numpy for display
    image = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image = (image * 0.5) + 0.5  # Undo normalization
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    # Overlay heatmap on the original image
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return superimposed_img