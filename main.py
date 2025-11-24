def main():
    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    from pathlib import Path
    import torch
    import torchvision
    import random
    from torch import nn
 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'you are using: {device}')

    weights = torchvision.models.AlexNet.DEFAULT
    auto_transforms = weights.transforms()

    image_path = Path('/content/drive/MyDrive/your_project/images')
    train_dir = image_path / 'tranidata/direction'
    test_dir = image_path / 'testdata/direction'

    # Add checks for directories
    assert train_dir.exists(), f"Train dir not found: {train_dir}"
    assert test_dir.exists(), f"Test dir not found: {test_dir}"

    import dataloader
    train_dataloader, test_dataloader, class_names = dataloader.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=auto_transforms,
        batch_size=32
    )

    # Build model with pretrained weights
    model = torchvision.models.AlexNet(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    output_shape = len(class_names)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=output_shape, bias=True)
    ).to(device)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    import Train_function
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from timeit import default_timer as timer
    start_time = timer()

    results = Train_function.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device
    )

    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    from helper_functions import plot_loss_curves
    plot_loss_curves(results)

    from helper_functions import pred_and_plot_image
    num_images_to_plot = 3
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
    test_image_path_sample = random.sample(test_image_path_list, k=num_images_to_plot)

    for image_path in test_image_path_sample:
        pred_and_plot_image(
            model=model,
            image_path=image_path,
            class_names=class_names,
            transform=weights.transforms(),
            image_size=(3, 224, 224),
        )

if __name__ == "__main__":
    main()