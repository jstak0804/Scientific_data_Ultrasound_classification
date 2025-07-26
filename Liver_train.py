import time
import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import datetime

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torchmetrics.classification import F1Score, Precision, Recall, ConfusionMatrix
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# 사용 중인 사용자 정의 함수 및 모델 import
from functions_for_train import find_image_sizes, EarlyStopping, calculate_optimal_size, get_unique_filename, save_results_to_csv
from model import EnhancedResNet, CustomModel
from Grad_Cam import log_gradcam_examples, log_gradcam_to_wandb, overlay_heatmap_on_image, generate_gradcam_heatmap, visualize_cam, show_cam_on_image, GradCAM

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()

# 랜덤 시드 고정 함수
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# ----------------------
# 결과 저장 폴더 생성 함수 (현재 날짜와 모델명을 이용)
def create_result_dir(model_name, base_dir="Result"):
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    result_dir = os.path.join(base_dir, date_str, model_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

# ----------------------
# Argument Parser Setup
random_seeds = random.sample(range(1, 3000), 20)

parser = argparse.ArgumentParser(description='Train a ResNet model with CBAM for ORS classification.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training, set to -1 for auto allocation')
parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights (.pth file)')
parser.add_argument('--seeds', nargs='+', type=int, default=[24], help='List of seeds to try for best result')
args = parser.parse_args()

# WandB 초기화
wandb.init(project="US_classification_seed_changing", entity="ddurbozak")
wandb.config.update({"learning_rate": 1e-5, "epochs": args.epochs, "batch_size": args.batch_size})

# 모델명을 지정하고 결과 폴더 생성 (원하는 모델 이름으로 수정 가능)
model_name = "CustomModel"
result_dir = create_result_dir(model_name)
print("Results will be saved in:", result_dir)

# ----------------------
# 데이터 전처리 및 로더 설정

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_path = "Liver_US/train_2"
test_path = "Liver_US/test_2"

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

batch_size = args.batch_size if args.batch_size != -1 else len(train_dataset) // 100

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
dataloaders = {'train': train_dataloader, 'test': test_dataloader}

# Metric 초기화 (테스트 시 사용)
precision_metric = Precision(num_classes=2, average='macro', task='multiclass').to(device)
recall_metric = Recall(num_classes=2, average='macro', task='multiclass').to(device)
confusion_metric = ConfusionMatrix(num_classes=2, task="multiclass").to(device)

# ----------------------
# 학습 함수 정의
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct_predictions = 0.0, 0
    f1_metric = F1Score(num_classes=2, average='macro', task='multiclass').to(device)
    precision_metric = Precision(num_classes=2, average='macro', task='multiclass').to(device)
    recall_metric = Recall(num_classes=2, average='macro', task='multiclass').to(device)

    for inputs, labels in tqdm(dataloader, desc="Training", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct_predictions += torch.sum(preds == labels.data).item()

        f1_metric.update(outputs, labels)
        precision_metric.update(outputs, labels)
        recall_metric.update(outputs, labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / len(dataloader.dataset)
    epoch_f1 = f1_metric.compute()
    epoch_precision = precision_metric.compute()
    epoch_recall = recall_metric.compute()

    wandb.log({
        "Train Loss": epoch_loss,
        "Train Accuracy": epoch_acc,
        "Train F1": epoch_f1.item(),
        "Train Precision": epoch_precision.item(),
        "Train Recall": epoch_recall.item()
    })

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

# ----------------------
# 테스트 함수 정의
def test(model, test_dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data).item()
            print(f"Batch Loss: {loss.item():.4f}, Precision: {precision_metric.compute():.4f}, Recall: {recall_metric.compute():.4f}")

    test_loss = running_loss / len(test_dataloader.dataset)
    test_acc = correct_predictions / len(test_dataloader.dataset)

    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Benign", "Malignant"])
    disp.plot(cmap="viridis")
    plt.pause(10)
    plt.close()

    wandb.log({
        "Test Loss": test_loss,
        "Test Accuracy": test_acc,
        "Test F1": test_f1,
        "Test Precision": test_precision,
        "Test Recall": test_recall
    })

    return test_loss, test_acc, test_f1, test_precision, test_recall

# ----------------------
# 학습과 평가를 진행하는 함수 (Epoch마다 Confusion Matrix 저장 포함)
def train_and_evaluate_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    results = []
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_test_acc, best_epoch_acc = 0.0, 0
    best_epoch_confusion_matrix = None
    best_f1, best_precision, best_recall = 0.0, 0.0, 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_metrics = {
            'train_loss': 0.0, 'train_acc': 0.0, 'train_f1': 0.0, 'train_precision': 0.0, 'train_recall': 0.0,
            'test_loss': 0.0, 'test_acc': 0.0, 'test_f1': 0.0, 'test_precision': 0.0, 'test_recall': 0.0
        }
        all_preds = []
        all_labels = []

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, correct_predictions = 0.0, 0
            f1_metric = F1Score(num_classes=2, average='macro', task='multiclass').to(device)
            precision_metric = Precision(num_classes=2, average='macro', task='multiclass').to(device)
            recall_metric = Recall(num_classes=2, average='macro', task='multiclass').to(device)
            all_preds_phase = []
            all_labels_phase = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    f1_metric.update(outputs, labels)
                    precision_metric.update(outputs, labels)
                    recall_metric.update(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    correct_predictions += torch.sum(preds == labels.data).item()

                    all_preds_phase.extend(preds.cpu().numpy())
                    all_labels_phase.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct_predictions / len(dataloaders[phase].dataset)
            epoch_f1 = f1_metric.compute()
            epoch_precision = precision_metric.compute()
            epoch_recall = recall_metric.compute()

            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc'] = epoch_acc
            epoch_metrics[f'{phase}_f1'] = epoch_f1.item()
            epoch_metrics[f'{phase}_precision'] = epoch_precision.item()
            epoch_metrics[f'{phase}_recall'] = epoch_recall.item()

            if phase == 'test':
                all_preds = all_preds_phase
                all_labels = all_labels_phase

                # Confusion Matrix 계산 및 플롯 저장
                cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
                print(f"Confusion Matrix for Epoch {epoch + 1}:\n", cm)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
                disp.plot(cmap=plt.cm.Blues)
                cm_filename = os.path.join(result_dir, f"confusion_matrix_epoch_{epoch + 1}.png")
                plt.savefig(cm_filename)
                wandb.log({f"Confusion Matrix Epoch {epoch + 1}": wandb.Image(cm_filename)})
                plt.close()

                # best model tracking
                if epoch_acc > best_test_acc:
                    best_test_acc = epoch_acc
                    best_epoch_acc = epoch + 1
                    best_epoch_confusion_matrix = cm
                    best_f1 = epoch_metrics['test_f1']
                    best_precision = epoch_metrics['test_precision']
                    best_recall = epoch_metrics['test_recall']

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, "
                  f"F1 Score: {epoch_f1:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}")
            wandb.log({
                f"{phase.capitalize()} Loss": epoch_loss,
                f"{phase.capitalize()} Accuracy": epoch_acc,
                f"{phase.capitalize()} F1": epoch_f1.item(),
                f"{phase.capitalize()} Precision": epoch_precision.item(),
                f"{phase.capitalize()} Recall": epoch_recall.item()
            })

        results.append([
            epoch + 1,
            epoch_metrics['train_loss'], epoch_metrics['train_acc'], epoch_metrics['train_f1'],
            epoch_metrics['train_precision'], epoch_metrics['train_recall'],
            epoch_metrics['test_loss'], epoch_metrics['test_acc'], epoch_metrics['test_f1'],
            epoch_metrics['test_precision'], epoch_metrics['test_recall']
        ])

        # 결과를 CSV 파일로 저장 (seed 별 결과로 저장)
        csv_filename = os.path.join(result_dir, f"results_seed_{seed}.csv")
        save_results_to_csv(results, csv_filename)

        if early_stopping(epoch_metrics['test_loss'], model):
            break

    print(f"\nBest Test Accuracy: {best_test_acc:.4f} at Epoch {best_epoch_acc}")
    print(f"Confusion Matrix for Best Epoch:\n{best_epoch_confusion_matrix}")
    return best_test_acc, best_epoch_acc, best_epoch_confusion_matrix, epoch_f1.item(), epoch_precision.item(), epoch_recall.item()

# ----------------------
# Confusion Matrix를 WandB에 로그하는 함수
def log_confusion_matrix_to_wandb(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="viridis", ax=ax)
    plt.title("Confusion Matrix")
    wandb.log({"Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)

# ----------------------
# Main loop: 여러 seed에 대해 학습 실행 및 결과 저장
if __name__ == "__main__":
    best_seed, best_accuracy = None, 0.0
    columns = ["Seed", "Test Accuracy", "Best Epoch", "Best F1 Score", "Test Precision", "Test Recall"]
    seed_results = []

    for seed in args.seeds:
        print(f"\nTraining with Seed: {seed}")
        set_seed(seed)
        model = CustomModel(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.NAdam(model.parameters(), lr=wandb.config["learning_rate"])

        test_accuracy, best_epoch_acc, best_epoch_confusion_matrix, best_f1, best_precision, best_recall = \
            train_and_evaluate_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs)

        seed_results.append([seed, test_accuracy, best_epoch_acc, best_f1, best_precision, best_recall])

    # Seed별 결과 출력 및 최종 CSV 저장
    for seed, test_acc, best_epoch, best_f1, test_precision, test_recall in seed_results:
        print(f"Seed: {seed}, Test Accuracy: {test_acc:.4f}, Best Epoch: {best_epoch}, "
              f"F1: {best_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

    results_df = pd.DataFrame(seed_results, columns=columns)
    overall_results_filename = os.path.join(result_dir, f"overall_seed_results_{time.strftime('%Y%m%d_%H%M%S_pararell')}.csv")
    results_df.to_csv(overall_results_filename, index=False)
    wandb.save(overall_results_filename)
    print("\nTraining complete. Results saved and logged to WandB.")
    wandb.finish()

    # (옵션) Best Seed의 모델에 대해 추가 테스트 및 Confusion Matrix 로그
    if best_seed is not None:
        print(f"\nLogging Confusion Matrix for Best Seed: {best_seed}")
        set_seed(best_seed)
        best_model = CustomModel(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.NAdam(best_model.parameters(), lr=wandb.config["learning_rate"])
        
        # 테스트 데이터로 Confusion Matrix 생성
        all_preds, all_labels = [], []
        best_model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc="Generating Confusion Matrix"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = best_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        log_confusion_matrix_to_wandb(all_labels, all_preds, class_names=["Benign", "Malignant"])
