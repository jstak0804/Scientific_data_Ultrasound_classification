# full_training_with_gradcam.py
import os, argparse, time, random
import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import timm
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, auc
from torchmetrics.classification import Precision, Recall, F1Score
import wandb

from Grad_Cam import GradCAM, show_cam_on_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_target_layer(model, model_name):
    if model_name.startswith("vgg"):
        return [model.features[-1]]
    elif model_name.startswith("resnet"):
        return [model.layer4[-1]]
    elif model_name.startswith("efficientnet"):
        return [model.features[-1]]
    elif "swin" in model_name:
        return [model.norm]
    else:
        raise ValueError(f"Unsupported model for GradCAM: {model_name}")

def get_model(model_name, num_classes):
    if model_name == "vgg16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "swin_tiny_patch4_window7_224":
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def generate_and_log_gradcam(model, dataloader, model_name, class_names, num_images=3):
    model.eval()
    target_layers = get_target_layer(model, model_name)
    gradcam = GradCAM(model=model, target_layers=target_layers)
    cnt = 0

    output_dir = os.path.join("gradcam_results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    for images, labels in dataloader:
        for i in range(images.size(0)):
            if cnt >= num_images:
                return
            input_tensor = images[i].unsqueeze(0).to(device)
            input_img_np = images[i].permute(1, 2, 0).cpu().numpy()
            input_img_np = (input_img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            input_img_np = np.clip(input_img_np, 0, 1)

            grayscale_cam = gradcam(input_tensor=input_tensor)[0]
            cam_img = show_cam_on_image(input_img_np, grayscale_cam, use_rgb=True)

            save_path = os.path.join(output_dir, f"gradcam_{model_name}_{cnt}.png")
            plt.imsave(save_path, cam_img)
            wandb.log({f"GradCAM_{model_name}_{cnt}": wandb.Image(save_path)})
            cnt += 1

def train_and_evaluate(model_name, train_path, test_path, seed=24, epochs=10, batch_size=32):
    set_seed(seed)
    wandb.init(project="MultiModel_US_Classification", name=f"{model_name}_seed{seed}")
    wandb.config.update({"model": model_name, "seed": seed, "epochs": epochs, "batch_size": batch_size})

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(train_path, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(model_name, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "Val F1": val_f1,
            "Val Precision": val_precision,
            "Val Recall": val_recall
        })

    # Final Test Evaluation
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    wandb.log({
        "Test Accuracy": acc,
        "Test F1": f1,
        "Test Precision": precision,
        "Test Recall": recall
    })

    # AUC-ROC 커브 저장
    from sklearn.preprocessing import label_binarize
    all_probs = np.array(all_probs)
    all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    output_dir = os.path.join("gradcam_results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"AUC-ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    roc_save_path = os.path.join(output_dir, f"roc_auc_{model_name}.png")
    plt.savefig(roc_save_path)
    plt.close()

    wandb.log({f"ROC_AUC_{model_name}": wandb.Image(roc_save_path)})

    generate_and_log_gradcam(model, test_loader, model_name, class_names)
    wandb.finish()

    return model_name, acc, f1, precision, recall

if __name__ == "__main__":
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()

    model_list = ["vgg16", "resnet50", "efficientnet_b0", "swin_tiny_patch4_window7_224"]
    summary = []

    for model_name in model_list:
        for seed in args.seeds:
            print(f"\n🧪 Running {model_name} | Seed: {seed}")
            result = train_and_evaluate(model_name, args.train_path, args.test_path, seed, args.epochs, args.batch_size)
            summary.append(result)

    import pandas as pd
    df = pd.DataFrame(summary, columns=["Model", "Accuracy", "F1", "Precision", "Recall"])
    filename = f"model_comparison_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print("📁 Results saved to", filename)
