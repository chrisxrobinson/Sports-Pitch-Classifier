from datasets import load_dataset

def load_resisc45_dataset():
    return load_dataset('timm/resisc45', split='train')

if __name__ == "__main__":
    dataset = load_resisc45_dataset()
    print(dataset)
