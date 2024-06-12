import numpy as np
import matplotlib.pyplot as plt
import sys


tri_cone_to_basis = np.array([
    [0.06679689867720229, -2.182544526081235, 2.8501615780317535 ],
    [-0.14050252150175271, 1.975556957351253, -0.9299812283485464],
    [0.9812095929616621, -0.14983895049667764, 0.07044113323759593]
])


# TODO: This array should be generated from (or maybe included) in tetra metadata.
# Temporarily copying this here as a global variable so that this script isn't super slow.
tetra_basis_to_cone = np.array([
    [1.024611997285031526417134473128e-05,
     2.242812871777761639813936200838e-04,
     8.319914273605166776803798711626e-02,
     1.025172264985438008721985170268e+00],
    [5.311379457600336889688819042021e-02,
     3.866034406628113262449630838091e-01,
     6.207455522878053688629051976022e-01,
     9.346738062565414228988203149129e-02],
    [1.403566167392598096341771451989e-01,
     4.705599399499938439994650707376e-01,
     5.023677978953242639903464805684e-01,
     7.081496318011284984983433332673e-02],
    [2.339692167721606763652886229465e-01,
     4.900264687105190808402710445080e-01,
     4.362919654283259340843414975097e-01,
     4.822367006387873883399564078900e-02]])


def convert_rg1g2b_rgb(tetra_data):
    """
    Convert RG1G2B to RGB and extract Q values.
    :param tetra_data: An array of shape (height, width, 4).
    """
    # Convert RG1G2B to LMSQ.
    h, w, ch = tetra_data.shape
    tetra_matrix = tetra_data.reshape(h * w, ch).T
    lmsq = tetra_basis_to_cone @ tetra_matrix

    # Drop Q values and convert to RGB.
    lms = lmsq[:-1, :]
    tri_matrix = tri_cone_to_basis @ lms
    rgb_data = (tri_matrix.T).reshape(h, w, 3)
    rgb_data = np.rint(rgb_data).astype(int)

    # Reshape Q values.
    q_values = lmsq[-1, :]
    q_data = q_values.reshape(h, w)

    return rgb_data, q_data


def split_tetra_data(tetra_data):
    """
    Return the R, G1, G2, and B channels individually.
    :param tetra_data: An array of shape (height, width, 4).
    """
    return tetra_data[:, :, 0], tetra_data[:, :, 1], tetra_data[:, :, 2], tetra_data[:, :, 3]


def view_tetra_data(tetra_data):
    """
    Use matplotlib.pytplot to visualize tetra_data.
    :param tetra_data: An array of shape (height, width, 4).
    """
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    axs = axs.flatten()

    # Display individual channels in grayscale.
    channels = split_tetra_data(tetra_data)
    channel_names = ['R', 'G1', 'G2', 'B']
    for i, ch in enumerate(channels):
        axs[i].imshow(ch, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'{channel_names[i]}')

    # Display RGB conversion and Q values.
    rgb_data, q_data = convert_rg1g2b_rgb(tetra_data)
    axs[5].imshow(rgb_data)
    axs[5].axis('off')
    axs[5].set_title('RGB')

    axs[6].imshow(q_data, cmap='gray')
    axs[6].axis('off')
    axs[6].set_title('Q values')

    fig.delaxes(axs[4])
    fig.delaxes(axs[7])
    plt.tight_layout()
    plt.show()


def view_tetra_file_on_click(file_path):
    """
    Use matplotlib.pyplot to visualize tetra_data as 4 grayscale
    images for individual R, G1, G2, B channels as well as
    converted to RGB with Q values.
    :param file_path: This script can be set up to run when
        clicking on a .npy file.
    """
    try:
        tetra_data = np.load(file_path)
    except Exception as e:
        print(f'Error loading {file_path}: {e}')

    # Display individual channels in grayscale.
    r, g1, g2, b = split_tetra_data(tetra_data)
    fig_gray, axs_gray = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    axs_gray = axs_gray.flatten()

    channel_names = ['R', 'G1', 'G2', 'B']
    channels = [r, g1, g2, b]

    for i, ch in enumerate(channels):
        axs_gray[i].imshow(ch, cmap='gray')
        axs_gray[i].axis('off')
        axs_gray[i].set_title(f'{channel_names[i]}')

    fig_gray.suptitle('RG1G2B Channels')
    plt.tight_layout()
    plt.show(block=False)

    # Display RGB conversion and Q values.
    fig_rgbq, axs_rgbq = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    axs_rgbq = axs_rgbq.flatten()

    rgb_data, q_data = convert_rg1g2b_rgb(tetra_data)
    axs_rgbq[0].imshow(rgb_data)
    axs_rgbq[0].axis('off')
    axs_rgbq[0].set_title('RGB')

    axs_rgbq[1].imshow(q_data, cmap='gray')
    axs_rgbq[1].axis('off')
    axs_rgbq[1].set_title('Q values')

    fig_rgbq.suptitle('RGB conversion and Q values')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    view_tetra_file_on_click(sys.argv[1])

