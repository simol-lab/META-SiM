"""Library for linear algebra operations in META-SiM."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Optional


def principal_project(embedding, kernel, effective_rank_fraction=0.25):
    """Projects the embeddings into the principal directions of kernel.

    This is the main function for principal projection. The kernel can
    be from a single downtream task or the concatenated kernel across
    multiple tasks.

    Args:
        embedding: The original embedding vectors as a numpy tensor.
        kernel: A numpy tensor of kernel.
        effective_rank_fraction: The fraction of 2nd order momentum to keep for low rank approximation.

    Returns:
        The projected embeddings.
    """


    u1, s1, _ = np.linalg.svd(kernel, full_matrices=False, hermitian=False)

    effective_rank_1 = u1.shape[1]
    for i in range(s1.shape[0]):
        if np.sum(s1[i:] ** 2) < (1.0 - effective_rank_fraction) * np.sum(s1 ** 2):
            effective_rank_1 = i
            break
    return np.einsum('ij,jk->ik', embedding, u1[:, :effective_rank_1])


def get_kernel_alignment(kernel_1, kernel_2, effective_rank_fraction=0.25):
    """Measures the alignment between two linear kernels.

    This function is used to measure the alignment between two tasks.

    Args:
        kernel_1: A numpy tensor of kernel 1.
        kernel_2: A numpy tensor of kernel 2.
        effective_rank_fraction: The fraction of 2nd order momentum to keep for low rank approximation.

    Returns:
        The alignment coefficient between [0, 1].
    """
    u1, s1, _ = np.linalg.svd(kernel_1, full_matrices=False, hermitian=False)
    u2, s2, _ = np.linalg.svd(kernel_2, full_matrices=False, hermitian=False)

    effective_rank_1 = u1.shape[1]
    for i in range(s1.shape[0]):
        if np.sum(s1[i:] ** 2) < (1.0 - effective_rank_fraction) * np.sum(s1 ** 2):
            effective_rank_1 = i
            break
    effective_rank_2 = u2.shape[1]
    for i in range(s2.shape[0]):
        if np.sum(s2[i:] ** 2) < (1.0 - effective_rank_fraction) * np.sum(s2 ** 2):
            effective_rank_2 = i
            break
    coeff = np.matmul(np.transpose(u1[:, :effective_rank_1]), u2[:, :effective_rank_2])
    return np.linalg.norm(coeff, 2)
