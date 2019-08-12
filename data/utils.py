import numpy as np
import random
import matplotlib.pyplot as plt


def _is_numpy_image(image):
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})


def crop(image, label, mask, size, coverage_ratio):
    if not _is_numpy_image(image):
        raise TypeError('image should be ndarray. Got {}.'.format(type(image)))
    if label == 0:
        pivot = tuple(map(lambda l: random.randint(0, l - 1), mask.shape))  # top-left
        h1, w1, _ = pivot
        h2 = h1 + size[0] - 1
        w2 = w1 + size[1] - 1
        h, w, _ = mask.shape
        if h2 > h - 1:
            delta = h2 - (h - 1)
            h1 -= delta
            h2 = h - 1
        if w2 > w - 1:
            delta = w2 - (w - 1)
            w1 -= delta
            w2 = w - 1
    else:
        coverage_pivot = tuple(
            map(lambda l: random.randint(int(l * (1 - coverage_ratio) / 2), int(l * (1 + coverage_ratio) / 2) - 1),
                size))
        indices = np.where(mask[:,:,1] == label)
        i = random.randint(0, len(indices[0]) - 1)
        defect_pivot = (indices[0][i], indices[1][i])
        h1 = defect_pivot[0] - coverage_pivot[0]
        h2 = h1 + size[0] - 1
        w1 = defect_pivot[1] - coverage_pivot[1]
        w2 = w1 + size[1] - 1
        h, w, _ = mask.shape
        if h1 < 0:
            delta = -h1
            h1 = 0
            h2 += delta
        if w1 < 0:
            delta = -w1
            w1 = 0
            w2 += delta
        if h2 > h - 1:
            delta = h2 - (h - 1)
            h1 -= delta
            h2 = h - 1
        if w2 > w - 1:
            delta = w2 - (w - 1)
            w1 -= delta
            w2 = w - 1
    image = image[h1:h2 + 1, w1:w2 + 1] if len(image.shape) == 2 else image[h1:h2 + 1, w1:w2 + 1, :]
    mask = mask[h1:h2 + 1, w1:w2 + 1, :]
    return np.ascontiguousarray(image), label, np.ascontiguousarray(mask)


def augment_img_mask(input_image, input_mask, augment=None):
    img, mask = input_image.copy(), input_mask.copy()

    if 'hflip' in augment and np.random.random() < 0.5:
        img = np.flip(img, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    if 'vflip' in augment and np.random.random() < 0.5:
        img = np.flip(img, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()

    if 'rot90' in augment and np.random.random() < 0.5:
        img = np.rot90(img, 1).copy()
        mask = np.rot90(mask, 1).copy()

    return img, mask


def visual_img_label(batch_img, batch_label, dtype='NCHW'):
    if dtype == 'NCHW':
        batch_img = np.transpose(batch_img, axes=[0, 2, 3, 1])
        batch_label = np.transpose(batch_label, axes=[0, 2, 3, 1])

    num_normal, num_defect = 0, 0
    batch_size = batch_img.shape[0]
    for i in range(batch_size):
        visual_x = batch_img[i, :, :, :]
        visual_x = np.asarray(visual_x, dtype=np.uint8)

        visual_y = batch_label[i, :, :, 1]
        visual_y = np.asarray(visual_y, dtype=np.uint8) * 255

        if np.where(visual_y == 255)[0].shape[0] > 0:
            num_defect += 1
        else:
            num_normal += 1

        fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4 * 1))

        _ = axes[0].imshow(visual_x)
        _ = axes[0].title.set_text('patch')
        _ = axes[1].imshow(visual_y, cmap='gray')
        _ = axes[1].title.set_text('mask')

        plt.show()
        plt.close()
    # print('Normal: {}, Defect: {}'.format(num_normal, num_defect))