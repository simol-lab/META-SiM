"""Library for model-based scores defined in the META-SiM paper."""

NN_K = 50  # how many nearest neighbors are considered for the score


def calculate_distance(embedding):
    """Calculates the distance matrix from the embeddings."""
    import tensorflow as tf
    with tf.device('/CPU:0'):
        distance = (
            tf.expand_dims(tf.einsum('ik,ik->i', embedding, embedding), axis=-1)
            + tf.expand_dims(tf.einsum('ik,ik->i', embedding, embedding), axis=0)
            - 2.0 * tf.einsum('ik,jk->ij', embedding, embedding))
    return distance


def get_entropy(embedding, label, branching_factor=NN_K):
    """Calculates the local Shannon entropy for traces.

    This function calculates the local Shannon entropy (LSE) defined in:
    https://doi.org/10.1101/2024.08.26.609721

    Args:
        embedding: The tensor of all embeddings.
        label: The list of all labels.
        branching_factor: The number of nearest neighbors used for calculation.

    Returns:
        A numpy array of entropy.
    """
    import numpy as np
    import tensorflow as tf
    distance = calculate_distance(embedding)
    top_k_results = tf.math.top_k(
        -distance, k=branching_factor + 1, sorted=True, name=None
    )
    top_k_idx = top_k_results.indices.numpy()
    n = embedding.shape[0]
    entropy = []
    for i in range(n):
        this_label = label[i]
        neighbour_label = label[top_k_idx[i, :].flatten()]
        unique_labels = set(neighbour_label)
        shannon_entropy = 0
        for l in unique_labels:
            p = np.mean(neighbour_label == l)
            shannon_entropy -= p * np.log(p)
        entropy.append(shannon_entropy)
    return np.array(entropy)


def get_self_consistency_score(embedding, label, target_label, branching_factor=NN_K):
    """Calculates the self-consistency score (LCS) defined in the META-SiM paper.

    This function calculated the self-consistency score (LCS) for labels, defined in:
    https://doi.org/10.1101/2024.08.26.609721

    LCS is a score for each unique value of label, and can be further averaged across
    all unique values of the labels.

    Args:
        embedding: The full embedding tensor for all traces.
        label: The full list of labels for all traces.
        target_label: The label for which the score is calculated.
        branching_factor: The number of nearest neighbors used for calculation.

    Returns:
        A numpy array of LCS values.
    """
    import numpy as np
    import tensorflow as tf

    distance = calculate_distance(embedding)
    top_k_results = tf.math.top_k(
        -distance, k=branching_factor+1, sorted=True, name=None
    )
    top_k_idx = top_k_results.indices.numpy()
    selected_labels = (label == target_label)
    top_k_labels = label[top_k_idx[selected_labels, 1:].flatten()]
    p_e = np.mean(label == target_label)
    return np.mean(top_k_labels == target_label)
