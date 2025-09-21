from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

class EuroSATDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset("MuafiraThasni/eurosat-dataset-with-image", split=split)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.in_c = 3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item['image']), item['label']

def build_eurosat_dataset(is_train=True, args=None):
    return EuroSATDataset('train' if is_train else 'validation')