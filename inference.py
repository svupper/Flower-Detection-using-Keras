import torch
from model import Resnet50Model
from data import FlowersDataModule
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class FlowerResnetPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = Resnet50Model.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = FlowersDataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = self.processor.image_cat

    def predict(self, sample_image):
        model_pred=self.model.predict(sample_image)
        predicted_class=self.lables[np.argmax(model_pred)]
        return predicted_class


if __name__ == "__main__":
    # height,width=180,180
    # flowers_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    # flowers_data = tf.keras.utils.get_file('flower_photos', origin=flowers_url, untar=True)
    # flowers_data = pathlib.Path(flowers_data)
    # all_sunflowers = list(flowers_data.glob('sunflowers/*'))
    # sample_image=cv2.imread(str(all_sunflowers[1]))
    # sample_image_resized= cv2.resize(sample_image, (height, width))
    # sample_image=np.expand_dims(sample_image_resized,axis=0)
    # print(sample_image.shape)
    data_dir = "flower_photos"
    transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(data_dir, transform=transform)
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])
    loader = DataLoader(train_data, batch_size=1, shuffle=True)
    image_to_predict = next(iter(loader))[0]
    predictor = FlowerResnetPredictor("./models/epoch=4-step=460.ckpt")
    print(predictor.predict(image_to_predict))
    plt.figure()
    plt.imshow(image_to_predict[0].permute(1, 2, 0))
    plt.show()