import matplotlib.pyplot as plt


def show_image(img_or_sample):
    if isinstance(img_or_sample, tuple):
        img = img_or_sample[0]
        label = img_or_sample[1]
    else:
        img = img_or_sample
        label = None
    img = img.cpu().numpy().transpose(1, 2, 0).squeeze()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img, cmap="gray_r")
    ax.axis("off")
    if label is not None:
        ax.set_title(f"label: {label}")
    plt.tight_layout()
    plt.show()
    return fig, ax
