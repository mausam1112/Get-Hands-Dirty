from math import e
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def explode(data):
    size = np.array(data.shape) * 2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def visualize_tensor(tensor: np.ndarray):
    """
    Visualizes a 2D or 3D tensor as a cuboid/cube using 3D voxels.

    Parameters:
    tensor (np.ndarray): Input tensor (2D or 3D)
    """
    # Check input dimensions
    if tensor.ndim not in [2, 3]:
        raise ValueError("Input must be a 2D or 3D tensor")

    # Convert 2D tensor to 3D by adding depth dimension
    is_2d = tensor.ndim == 2
    if is_2d:
        tensor = tensor[:, :, np.newaxis]

    shape = tensor.shape

    # Create coordinate grids for voxel edges
    x, y, z = np.indices(tuple(np.array(tensor.shape) + 1))

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = np.empty(tensor.shape, dtype=object)

    # colors[:, :, 0] = "red"
    # colors[:, :, 1] = "yellow"
    # colors[:, :, 2] = "blue"
    # colors[:, :, 3] = "green"
    color_map = [
        "red",
        "yellow",
        "blue",
        "green",
        "cyan",
        "magenta",
        "orange",
        "purple",
    ]
    for i in range(shape[2]):  # Depth
        colors[:, :, i] = color_map[i % len(color_map)]  # Cycle through colors

    # Plot voxels with face colors
    ax.voxels(x, y, z, tensor, facecolors=colors, edgecolors="gray", alpha=0.5)

    # Add text (numbers) on the voxel faces
    for x_idx in range(shape[0]):  # Column
        for y_idx in range(shape[1]):  # Row
            for z_idx in range(shape[2]):  # Depth
                if z_idx > 0:
                    continue
                value = tensor[x_idx, y_idx, z_idx]
                label = f"{value:.2f}" if isinstance(value, float) else f"{value}"
                # bg_color = colors[x_idx, y_idx, z_idx]
                # brightness = np.dot(bg_color[:3], [0.299, 0.587, 0.114])
                # text_color = "white" if brightness < 0.5 else "black"
                ax.text(
                    x_idx + 0.5,
                    y_idx + 0.5,
                    z_idx + 0.0,
                    label,
                    color="black",
                    fontsize=10,
                    ha="center",
                    va="center",
                    weight="bold",
                )
    # for x_idx in range(shape[0]):  # Column
    #     for z_idx in range(shape[2]):  # Depth
    #         for y_idx in range(shape[1]):  # Row
    #             if y_idx > 0:
    #                 continue
    #             value = tensor[x_idx, y_idx, z_idx]
    #             label = f"{value:.2f}" if isinstance(value, float) else f"{value}"
    #             # bg_color = colors[x_idx, y_idx, z_idx]
    #             # brightness = np.dot(bg_color[:3], [0.299, 0.587, 0.114])
    #             # text_color = "white" if brightness < 0.5 else "black"
    #             ax.text(
    #                 x_idx + 0.5,
    #                 y_idx + 0.5,
    #                 z_idx + 0.5,
    #                 label,
    #                 color="black",
    #                 fontsize=10,
    #                 ha="center",
    #                 va="center",
    #                 weight="bold",
    #             )
    # for z_idx in range(shape[2]):  # Depth
    #     for y_idx in range(shape[1]):  # Row
    #         for x_idx in range(shape[0]):  # Column
    #             if x_idx > 0:
    #                 continue
    #             value = tensor[x_idx, y_idx, z_idx]
    #             label = f"{value:.2f}" if isinstance(value, float) else f"{value}"
    #             # bg_color = colors[x_idx, y_idx, z_idx]
    #             # brightness = np.dot(bg_color[:3], [0.299, 0.587, 0.114])
    #             # text_color = "white" if brightness < 0.5 else "black"
    #             ax.text(
    #                 x_idx + 0.5,
    #                 y_idx + 0.5,
    #                 z_idx + 0.5,
    #                 label,
    #                 color="black",
    #                 fontsize=10,
    #                 ha="center",
    #                 va="center",
    #                 weight="bold",
    #             )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        f"{tensor.ndim}D Tensor Visualization"
        + (" (2D as Cuboid)" if is_2d else " (3D)")
    )

    ax.set_aspect("equal")

    # Adjust viewing angle for better visibility
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Create sample tensors
    matrix_2d = np.random.rand(5, 6)  # 2D tensor (5x6)
    tensor_3d = np.random.rand(4, 4, 5)  # 3D tensor (3x4x2)

    # Visualize the tensors
    # visualize_tensor(matrix_2d)
    visualize_tensor(tensor_3d)
