#%%
import numpy as np
import matplotlib.pyplot as plt

k = 6000
c = 0.85

data = np.genfromtxt("ICOMP_DATA/new iq0.3/2_10_6_12_6_0.7.txt")#ICOMP_DATA/7_15_6_12_6.txt")
q = data[:117,0]
I = data[:117,1]
# dq = 0.0019903845283018866
# qmin = 0.003
# qmax = 0.36
# n = int((qmax-qmin)/dq)+1
# q = np.array(range(n))*dq+qmin
# #q = np.linspace(qmin, qmax, num = n)
# q = np.round(q, 8)

#%%

# Supongo que I es un array o vector previamente definido.
# La siguiente línea parece ser una operación que normaliza los elementos de I.
I = (100 * I) / I[0]

# Buscar el índice del valor más cercano a 0.2 en el array q.
idx = np.abs(0.2 - q).argmin()

# Obtener el valor correspondiente en el array I usando el índice encontrado.
Iarb = I[idx]

# Calcular la desviación estándar 's' utilizando la fórmula dada.
s = np.sqrt((1 / (k * q)) * (I + (2 * c * Iarb) / (1 - c)))

# Generar un array 'Ie' de números aleatorios distribuidos normalmente
# con media 'I' y desviación estándar 's'.
Ie = np.random.normal(I, s)

figsize = (4,4)
fig, ax = plt.subplots(figsize=(figsize))
ax.plot(q,I,color='k',linestyle='-',ms=8,linewidth=1.3,marker=',',label=f'clean')
ax.plot(q,Ie,linestyle='-',ms=8,linewidth=1,marker='o',label='noised')
plt.xlim(q[0],q[-1])
plt.ylim(2*10**(-5),20)
plt.xlabel(r'q, $\AA^{-1}$',fontsize=20)
plt.ylabel(r'$I$(q)',fontsize=20)
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig('testplot.png',dpi=169,bbox_inches='tight')
plt.grid()
plt.show()
