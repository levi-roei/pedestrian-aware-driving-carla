import numpy as np


def clip_and_int_boxes(boxes: np.ndarray,
                       image_w: int,
                       image_h: int) -> np.ndarray:
    """
    Converts bounding box coordinates to integers and clips them to valid image bounds.

    Parameters:
        boxes (np.ndarray): An (N, 4) NumPy array of float bounding boxes in [x1, y1, x2, y2] format.
        image_w (int): Width of the image.
        image_h (int): Height of the image.

    Returns:
        np.ndarray: An (N, 4) array of integer bounding boxes clipped to lie within
                    [0, image_w - 1] for x-coordinates and [0, image_h - 1] for y-coordinates.
    """
    boxes = boxes.astype(int)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_w - 1)  # x1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_w - 1)  # x2
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_h - 1)  # y1
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_h - 1)  # y2
    return boxes
