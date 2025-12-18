import numpy as np
import matplotlib.pyplot as plt


'''
https://doi.org/10.1063/1.1668320 (König, 2004) - Updated ny for EPM
https://doi.org/10.1364/AO.42.006661 (Emanueli, 2003) - Updated temperature
https://doi.org/10.1364/AO.41.005040 (Kato, 2002) - Introduced temperature
https://doi.org/10.1063/1.123408 (Fradkin, 1999) - Updated nz
https://doi.org/10.1364/AO.26.002390 (Fan, 1987) - nx, ny, nz
'''

def ny(labda, T=25):
    # A, B, C, D = 2.19229, 0.83547, 0.04970, 0.01621
    A, B, C, D = 2.09930, 0.922683, 0.0467695, 0.0138408
    a =  6.2897 + 6.3061*labda - 6.0629*labda**2 + 2.6486*labda**3
    b = -0.14445 + 2.2244*labda - 3.5770*labda**2 + 1.3470*labda**3

    n25 = np.sqrt(A + B / (1 - C / labda ** 2) - D * labda ** 2)
    dn = a*(T - 25) + b*(T - 25)**2
    return n25 + dn * 1e-6


def nz(labda, T=25):
    # A, B, C, D = 2.19229, 0.83547, 0.04970, 0.01621
    A, B, C, D, E, F = 2.12725, 1.18431, 5.14852e-2, 0.6603, 100.00507, 9.68956e-3
    a = 9.9587 + 9.9228*labda - 8.9603*labda**2 + 4.1010*labda**3
    b = -1.1882 + 10.459*labda - 9.8136*labda**2 + 3.1481*labda**3

    # n25 =  np.sqrt(A + B/(1-C/labda**2) - D*labda**2)
    n25 = np.sqrt(A + B/(1-C/labda**2) + D/(1-E/labda**2) - F*labda**2)
    dn =  a*(T - 25) + b*(T - 25)**2
    return n25 + dn * 1e-6

def ny_g(labda, T=25):
    n = ny(labda, T)
    d_labda = 1e-9
    slope = (ny(labda + d_labda, T) - ny(labda, T)) / d_labda
    return n - labda * slope

def nz_g(labda, T=25):
    n = nz(labda, T)
    d_labda = 1e-9
    slope = (nz(labda + d_labda, T) - nz(labda, T)) / d_labda
    return n - labda * slope

T = 21
# pump
print(f'phase velocity pump: {ny(0.775, T):.3f}')
print(f'group velocity pump: {ny_g(0.775, T):.3f}')
print('\n')

# signal
print(f'phase velocity signal: {ny(1.55, T):.3f}')
print(f'group velocity signal: {ny_g(1.55, T):.3f}')
print('\n')

# idler
print(f'phase velocity idler: {nz(1.55, T):.3f}')
print(f'group velocity idler: {nz_g(1.55, T):.3f}')
print('\n')

kp = 2*np.pi*ny(0.775, T)/0.775
ks = 2*np.pi*ny(1.55, T)/1.55
ki = 2*np.pi*nz(1.55, T)/1.55

phase_mismatch = kp - ks - ki
poling = 2*np.pi / phase_mismatch
print(f'required poling: {abs(poling):.1f}um')

ngp = ny_g(0.775, T)
ngs = ny_g(1.55, T)
ngi = nz_g(1.55, T)

print(f'mismatch group velocity pump-signal: {ngp - ngs:.4f}')
print(f'mismatch group velocity pump-idler: {ngp - ngi:.4f}')

kp_prime = ngp
ks_prime = ngs
ki_prime = ngi

group_velocity_mismatch = 2*kp_prime - ks_prime - ki_prime
print(f'Group velocity mismatch: {abs(group_velocity_mismatch):.5f}')


'''
Script Output:

phase velocity pump: 1.758
group velocity pump: 1.811

phase velocity signal: 1.734
group velocity signal: 1.764

phase velocity idler: 1.816
group velocity idler: 1.853

required poling: 46.2um
mismatch group velocity pump-signal: 0.0471
mismatch group velocity pump-idler: -0.0415
Group velocity mismatch: 0.00551
'''