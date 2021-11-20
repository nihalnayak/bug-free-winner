from sklearn.decomposition import TruncatedSVD, PCA, SparsePCA


def run_decomposition(algo, data1, data2, components):
    print('using algorithm: ', algo)
    print('reducing dimensionality: mod1')
    # Do PCA on the input data
    embedder_mod1 = algo(n_components=components)
    x = embedder_mod1.fit_transform(data1)

    print('reducing dimensionality: mod2')
    embedder_mod2 = algo(n_components=components)
    y = embedder_mod2.fit_transform(data2)
    return x, y
