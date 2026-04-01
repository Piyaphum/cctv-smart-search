"""
AI Models Loading and Management
"""
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
from config import YOLO_MODEL


@st.cache_resource
def load_detection_model():
    """Load YOLO object detector"""
    return YOLO(YOLO_MODEL)


@st.cache_resource
def load_reid_model():
    """Load ResNet50 for person re-identification"""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


@st.cache_resource
def load_transforms():
    """Load image transformation pipelines"""
    base_transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    aug_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=10),
    ])
    return base_transform, aug_transform





def get_all_models():
    """Convenience function to load all models at once"""
    detector = load_detection_model()
    reid_model = load_reid_model()
    base_tf, aug_tf = load_transforms()
    return {
        'detector': detector,
        'reid_model': reid_model,
        'base_transform': base_tf,
        'aug_transform': aug_tf
    }
