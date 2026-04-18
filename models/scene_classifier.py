import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import pickle


class SceneClassifier(nn.Module):

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(SceneClassifier, self).__init__()

        # load mobilenetv2 with imagenet weights
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)

        # swap out final layer for our 4-class head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes
        self.class_names = ["road", "park", "street", "indoor"][:num_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SceneClassificationPipeline:

    SCENE_CLASSES = {
        0: "road",
        1: "park",
        2: "street",
        3: "indoor"
    }

    def __init__(
        self,
        device: str = "cpu",
        model_path: Path = None,
        num_classes: int = 4
    ):
        self.device = device
        self.num_classes = num_classes

        # init the model
        self.model = SceneClassifier(
            num_classes=num_classes,
            pretrained=True
        )

        # load finetuned weights if given
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()

        # todo: karan fix transform to support higher res input
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(
        self,
        image: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        # bgr to rgb
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess
        x = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)

        pred_class_id = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class_id].item()

        prob_dict = {
            self.SCENE_CLASSES.get(i, f"class_{i}"): probs[0, i].item()
            for i in range(self.num_classes)
        }

        predicted_class = self.SCENE_CLASSES.get(pred_class_id, "unknown")

        return predicted_class, confidence, prob_dict

    def save(self, model_path: Path):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self


class TrainingUtils:

    @staticmethod
    def create_train_dataloader(
        image_dir: Path,
        annotations_file: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True
    ):
        from torch.utils.data import DataLoader, Dataset

        class SceneDataset(Dataset):
            def __init__(self, image_dir: Path, annotations_file: Path):
                self.image_dir = Path(image_dir)
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                self.class_to_id = {
                    "road": 0, "park": 1, "street": 2, "indoor": 3
                }

                # load annotations csv
                self.samples = []
                with open(annotations_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            path, class_name = line.strip().split(',')
                            if class_name in self.class_to_id:
                                self.samples.append((path, class_name))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, class_name = self.samples[idx]
                img = cv2.imread(str(self.image_dir / img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb)
                class_id = self.class_to_id[class_name]
                return img_tensor, class_id

        dataset = SceneDataset(image_dir, annotations_file)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

    @staticmethod
    def train_epoch(
        model: SceneClassifier,
        dataloader,
        optimizer,
        criterion,
        device: str
    ) -> float:
        model.train()
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @staticmethod
    def evaluate(
        model: SceneClassifier,
        dataloader,
        criterion,
        device: str
    ) -> Tuple[float, float]:
        # todo: mudit fix evaluate to also return per-class accuracy
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy
