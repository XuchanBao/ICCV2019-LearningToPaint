import torch
import numpy as np

DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


coord = torch.zeros([1, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH])
for i in range(DEFAULT_HEIGHT):
    for j in range(DEFAULT_WIDTH):
        coord[0, 0, i, j] = i / (DEFAULT_HEIGHT - 1.)
        coord[0, 1, i, j] = j / (DEFAULT_WIDTH - 1.)
coord = coord.to(device)


def decode(delta_params, curr_params):
    """
    Transition: (delta_params, curr_params) -> new canvas after action x
    :param delta_params: action, i.e. delta parameters, shape = (N, frameskip (1) * action_dim (2))
    :param partial_state: (canvas, curr_params), shape = (N, 6, 128, 128)
        - canvas shape = (N, 3, 128, 128)
        - current parameter shape = (N, 2, 128, 128)
        - timestep shape = (N, 1, 128, 128)
    :return: new partial state, shape = (N, 6, 128, 128)
    """
    expanded_x = delta_params.unsqueeze(-1).unsqueeze(-1).expand_as(curr_params)
    next_params = curr_params + expanded_x  # shape = (N, 2, 128, 128)

    next_canvas, _, __ = generate_quadratic_heatmap(next_params[:, :, 0, 0])  # shape = (N, 3, 128, 128)

    next_canvas = next_canvas.reshape(-1, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH)
    normalized_next_canvas = next_canvas.float() / 255.0

    return normalized_next_canvas, next_params


def generate_quadratic_heatmap(batch_parameters, img_height=DEFAULT_HEIGHT, img_width=DEFAULT_WIDTH, return_params=False):
    """
    Generate quadratic heatmap z = (x - centre_x)^2 + (y - centre_y)^2
    :param batch_parameters: shape = (N, 2). Batched (centre_x, centre_y)
        - centre_x: float, in range [0, img_height)
        - centre_y: float, in range [0, img_width)
    :param img_height: integer
    :param img_width: integer
    :param return_params: boolean, whether or not return the centre used to generate data
    :return: numpy array, shape [img_height, img_width, 3 (channels)],
    optinal return: centre_x and centre_y
    """
    # if batch_parameters is None:
    #     centre_x, centre_y = get_initialization(img_height, img_width)

    batch_size = batch_parameters.shape[0]
    centre_x = batch_parameters[:, 0].unsqueeze(-1).unsqueeze(-1).expand(batch_size, img_height, img_width)
    centre_y = batch_parameters[:, 1].unsqueeze(-1).unsqueeze(-1).expand(batch_size, img_height, img_width)

    # arr = torch.zeros([batch_size, img_height, img_width])
    expanded_coord_x = coord[:, 0, :, :].expand(batch_size, img_height, img_width)
    expanded_coord_y = coord[:, 1, :, :].expand(batch_size, img_height, img_width)

    # Compute heat map
    arr = (expanded_coord_x - centre_x) ** 2 + (expanded_coord_y - centre_y) ** 2

    # Cast to uint8
    arr = (arr * 255).byte()

    # Copy the array across all three channels
    arr = arr.unsqueeze(-1).expand(batch_size, img_height, img_width, 3)

    if return_params:
        return arr, centre_x, centre_y

    return arr, None, None


def get_initialization(batch_size, img_height=DEFAULT_HEIGHT, img_width=DEFAULT_WIDTH):
    centre_x = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size, 1))).float()
    centre_y = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size, 1))).float()
    initial_params = torch.cat((centre_x, centre_y), 1)
    return initial_params
