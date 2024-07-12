from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from data_factory.data_loader import get_loader_segment
from utils import load_prototype_features
from model.prvae import PRVAE
import os

def test(class_name, batch_size=32, win_size=100, dataset="SMD", data_path="datasets"):
    test_loader = get_loader_segment(data_path, batch_size=batch_size, win_size=win_size, 
                                                                                mode="test", dataset=dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proto_features = load_prototype_features('prototypes', dataset, device)
    model = PRVAE("resnet18", num_classes=1, device=device).to(device)
    model.load_state_dict(
            torch.load(
                os.path.join('trained_models', str(dataset) + f'_checkpoint_gpu.pth')))
    model.eval()
    data_num = 0
    right_pred_num = 0
    pred_labels = []
    for i, (input_data, labels) in enumerate(test_loader):
        input = input_data
        input = np.expand_dims(input, axis=-1)
        input = np.repeat(input, repeats=3, axis=-1)
        input = np.transpose(input, (0, 3, 1, 2))
        input = input.to(device)

        output = model(input,proto_features)
        output = output.permute(0, 2, 3, 1)
        output = torch.mean(output, dim=-1, keepdim=True)
        output = output.squeeze(-1)

        output = output.detach().cpu().numpy()

        num, pred_label, right_num = right_score(output,input_data,labels=labels)
        data_num+=num
        pred_labels.append(pred_label.tolist())
        right_pred_num+=right_num
    
    print(f"data_num = {data_num}, right_pred_num = {right_pred_num}, accuracy = {right_pred_num / data_num}")
        


def right_score(output,input,labels,thrd=5):
    pred_label = np.abs(output-input)
    pred_label = pred_label.sum(axis=2)
    pred_label = np.where(pred_label > thrd,1,0)
    right_num = (pred_label==labels)
    return pred_label.size, pred_label, right_num


if __name__ == '__main__':
    test(class_name="SMD", batch_size=32, win_size=100, dataset="SMD", data_path="datasets")