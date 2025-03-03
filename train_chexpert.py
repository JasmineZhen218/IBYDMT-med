import os
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip
from tqdm import tqdm

import configs
import datasets
import models
from ibydmt import get_config, get_model, get_dataset, Constants as c

device = c.DEVICE
workdir = c.WORKDIR

config_name = "chexpert"
config = get_config(config_name)

model = get_model(config, device=device)
preprocess = model.preprocess
train_preprocess = Compose(
    [preprocess, RandomRotation(degrees=15), RandomHorizontalFlip(p=0.5)]
)

train_dataset = get_dataset(config, train=True, transform=train_preprocess)
train_dataset.load_image = True
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = get_dataset(config, train=False, transform=preprocess)
test_dataset.load_image = True
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
dataloaders = {"train": train_dataloader, "test": test_dataloader}

optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=1e-05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

wandb.init(project="chexpert")

best_accuracy = 0.0
for epoch in range(20):
    for op, dataloader in dataloaders.items():
        if op == "train":
            torch.set_grad_enabled(True)
            model.train()
        else:
            torch.set_grad_enabled(False)
            model.eval()

        running_samples = 0
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(tqdm(dataloader)):
            image, target = data

            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(image.expand(-1, 3, -1, -1)).squeeze()
            loss = F.binary_cross_entropy_with_logits(
                output, target.float(), reduction="sum"
            )

            if op == "train":
                loss.backward()
                optimizer.step()

            prediction = (output > 0).float()
            accuracy = (prediction == target).float().sum()

            running_samples += image.size(0)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

            log_step = 20
            if op == "train" and (i + 1) % log_step == 0:
                wandb.log(
                    {
                        f"{op}_loss": running_loss / running_samples,
                        f"{op}_accuracy": running_accuracy / running_samples,
                    }
                )
                running_samples = 0
                running_loss = 0.0
                running_accuracy = 0.0

        if op == "train":
            scheduler.step()
        if op == "test":
            wandb.log(
                {
                    f"{op}_loss": running_loss / running_samples,
                    f"{op}_accuracy": running_accuracy / running_samples,
                }
            )

            test_accuracy = running_accuracy / running_samples
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(
                    model.state_dict(),
                    os.path.join(workdir, "weights", "chexpert", "resnet.pt"),
                )
