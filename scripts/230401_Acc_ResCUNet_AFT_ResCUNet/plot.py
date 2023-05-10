import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

def plot(img: ndarray, masked_img: ndarray, predict_img: ndarray, input_metrics: dict[str, float], pred_metrics: dict[str, float]):
    max_value = img.max()
    img /= max_value
    masked_img /= max_value
    predict_img /= max_value

    if img.ndim == 3:
        selected_slc = img.shape[0] // 2
        img = img[selected_slc]
        masked_img = masked_img[selected_slc]
        predict_img = predict_img[selected_slc]

    input_res = np.abs(img - masked_img)
    predicted_res = np.abs(img - predict_img)
    fig, axs = plt.subplots(2, 3)

    axs[0][0].imshow(img, cmap='gray', vmin=0, vmax=np.max(img))
    axs[0][0].set_title(f'Target')

    axs[0][1].imshow(masked_img, cmap='gray', vmin=0, vmax=np.max(img))
    axs[0][1].set_title(f'Input')
    s = ''
    for k, v in input_metrics.items():
        s += f'{k} = {v:.5f}\n'
    s = s[:-1]
    axs[0][1].text(.05, .95, s, backgroundcolor='white',
                   transform=axs[0][1].transAxes, ha="left", va="top")

    axs[0][2].imshow(predict_img, cmap='gray', vmin=0, vmax=np.max(img))
    axs[0][2].set_title(f'Prediction')
    s = ''
    for k, v in pred_metrics.items():
        s += f'{k} = {v:.5f}\n'
    s = s[:-1]
    axs[0][2].text(.05, .95, s, backgroundcolor='white',
                   transform=axs[0][2].transAxes, ha="left", va="top")

    axs[1][0].remove()

    axs[1][1].imshow(input_res, cmap='hot', vmin=0, vmax=np.max(input_res))
    fig.colorbar(axs[1][1].get_images()[0], ax=axs[1][1])
    axs[1][2].imshow(predicted_res, cmap='hot', vmin=0, vmax=np.max(input_res))
    fig.colorbar(axs[1][2].get_images()[0], ax=axs[1][2])
    return fig
