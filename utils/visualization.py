import cv2 as cv
from scipy.ndimage.filters import gaussian_filter
import torch
import numpy as np
import matplotlib.pyplot as plt

def prepare_img(img, denormalize=None):
    img_array = img.detach().numpy().copy()
    img_array = img_array.transpose(1, 2, 0)
    img_array = denormalize(img_array) if denormalize != None else img_array
    img_array = (img_array * 255).astype(np.uint8)
    return img_array

def show_img(img, denormalize=None):
    img_array = prepare_img(img, denormalize)
    plt.imshow(img_array)
    plt.show()

def show_img_with_pos2d(img, pos2d, denormalize=None):
    img_array = prepare_img(img, denormalize)

    for pt in pos2d:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(img_array, pt, radius=0, color=(0, 255, 0), thickness=5)

    plt.imshow(img_array)
    plt.show()

def project_pos3d_to_pos2d(pos3d, project_matrix):
    pos3d_homo = np.concatenate((pos3d, np.ones((pos3d.shape[0], 1))), axis=-1)
    pos2d_homo = pos3d_homo @ project_matrix.T
    pos2d = pos2d_homo[:,:2] / pos2d_homo[:,2:3]
    return pos2d

def show_img_with_pos3d(img, pos3d, project_matrix, denormalize=None):
    pos2d = project_pos3d_to_pos2d(pos3d, project_matrix)
    show_img_with_pos2d(img, pos2d, denormalize)

def show_pos2d_with_projected_pos3d(pos2d, pos3d, canvas_size, project_matrix):
    canvas_pos2d = np.zeros(canvas_size)
    canvas_pos3d = np.zeros(canvas_size)

    for pt in pos2d:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(canvas_pos2d, pt, radius=0, color=(255, 255, 255), thickness=10)
    projected_pos3d = project_pos3d_to_pos2d(pos3d, project_matrix)
    for pt in projected_pos3d:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(canvas_pos3d, pt, radius=0, color=(255, 255, 255), thickness=10)

    ig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].imshow(canvas_pos2d)
    axes[1].imshow(canvas_pos3d)
    plt.show()

def show_pos3d(pos3d, edges):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2])
    for edge in edges:
        data = np.stack((np.array(pos3d[edge[0]]), np.array(pos3d[edge[1]])))
        ax.plot3D(data[:,0], data[:,1], data[:,2], 'gray')
    plt.show()

def show_img_with_hmap(img, hmap, denormalize=None):
    img_array = prepare_img(img, denormalize)
    hmap_array = hmap.numpy().sum(axis=0)

    ig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].imshow(img_array)
    axes[1].imshow(hmap_array)
    plt.show()

def pos2d_to_hmap(pos2d, img_size, downsample, sigma):
    hmap = np.zeros((img_size[0], img_size[1], pos2d.shape[0]), dtype=float)
    for i in range(0, pos2d.shape[0]):
        hmap[int(pos2d[i,1]), int(pos2d[i,0]), i] = 1.0
        hmap[:, :, i] = gaussian_filter(hmap[:, :, i], sigma=sigma)
    hmap[hmap > 1] = 1
    hmap[hmap < 0.001] = 0
    hmap = cv.resize(hmap, (img_size[0]//downsample, img_size[1]//downsample), cv.INTER_LINEAR)
    hmap = hmap.transpose(2, 0, 1)
    return hmap