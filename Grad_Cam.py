# file: Grad_Cam.py
import cv2
import numpy as np
torch = __import__('torch')
nn = __import__('torch.nn')

class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model.eval()
        self.target_layers = target_layers
        self.gradients = []

        for layer in self.target_layers:
            layer.register_forward_hook(self.forward_hook)
            layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor):
        output = self.model(input_tensor)
        class_idx = output.argmax().item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1

        return [cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[3]))]

def show_cam_on_image(img, mask, use_rgb=False):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    if use_rgb:
        heatmap = heatmap[..., ::-1]
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
