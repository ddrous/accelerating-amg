import pyamg


size = 2
def create_dataset(size=10, dist='lognormal'):
    if dist=='poisson':
        A = pyamg.gallery.poisson((size, size), dtype='float32', type='FE')
    elif dist=='lognormal':
        pass
    return A

A = create_dataset(dist='poisson')
print(A.todense())