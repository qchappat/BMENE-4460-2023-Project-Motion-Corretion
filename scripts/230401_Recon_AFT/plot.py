import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def plot(img: ndarray, pred_img: ndarray, pred_metrics: dict[str, float]):
    max_value = img.max()
    img /= max_value
    pred_img /= max_value

    if img.ndim == 3:
        selected_slc = img.shape[0] // 2
        img = img[selected_slc]
        pred_img = pred_img[selected_slc]

    predicted_res = np.abs(img - pred_img)
    fig, axs = plt.subplots(2, 2)

    axs[0][0].imshow(img, cmap='gray', vmin=0, vmax=np.max(img))
    axs[0][0].set_title(f'Target')

    axs[0][1].imshow(pred_img, cmap='gray', vmin=0, vmax=np.max(img))
    axs[0][1].set_title(f'Prediction')
    s = ''
    for k, v in pred_metrics.items():
        s += f'{k} = {v:.5f}\n'
    s = s[:-1]
    axs[0][1].text(.05, .95, s, backgroundcolor='white',
                   transform=axs[0][1].transAxes, ha="left", va="top")

    axs[1][0].remove()

    axs[1][1].imshow(predicted_res, cmap='hot', vmin=0, vmax=np.max(predicted_res))
    fig.colorbar(axs[1][1].get_images()[0], ax=axs[1][1])
    return fig
