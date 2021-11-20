def run_decomposition(algo, data1, data2, mod1_dim, mod2_dim):
    """Wrapper to reduce the dimensions of model.

    Args:
        algo (object): dimensionality reduction method
        data1 (object): modality 1
        data2 (object): modality 2
        mod1_dim (int): dimension of the first modalit
        mod2_dim (int): dimension of the second modality

    Returns:
        tuple: tuple containing the reduced dimensions.
    """
    print("using algorithm: ", algo)
    print("reducing dimensionality: mod1")
    # Do PCA on the input data
    embedder_mod1 = algo(n_components=mod1_dim)
    x = embedder_mod1.fit_transform(data1)

    print("reducing dimensionality: mod2")
    embedder_mod2 = algo(n_components=mod2_dim)
    y = embedder_mod2.fit_transform(data2)
    return x, y
