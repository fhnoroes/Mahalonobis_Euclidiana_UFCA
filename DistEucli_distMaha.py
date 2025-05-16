from scipy.spatial.distance import euclidean, mahalanobis
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

n = [10,20,40,80,100,500,1000]
distE = []
distMaha = []
distMahaI = []
beta0 = []
beta1 = []

#caso uma variável explicativa
for i in n:
    x = np.random.normal(10,2,i)
    y =  np.random.normal(15,2,i)
    #pega distância euclidiana
    distE.append(euclidean(x,y))
    #pega valores de x e y
    dados = []
    dados.append(x)
    dados.append(y)
    #pega mahalanobis com a matriz identidade
    mi = np.linalg.inv(np.eye(len(dados[0])))
    distMahaI.append(mahalanobis(x,y,mi))
    #pega matriz de covariância e a inverte
    covm = np.cov(dados,rowvar=False)
    incov = np.linalg.pinv(covm)
    #pega distancia mahalanobis
    distMaha.append(mahalanobis(x,y,incov))
    #adiciona os 1s ao x
    novox = sm.add_constant(x)
    modelo = sm.OLS(y,novox).fit()
    #pega beta0 e beta1
    beta0.append(modelo.params[0])
    beta1.append(modelo.params[1])

#plota os gráficos
plt.figure(figsize=(8,6))
plt.title("Gráfico dist. Euclidiana por Mahalanobis")
plt.xlabel("Mahalanobis")
plt.ylabel("dist. Euclidiana")
plt.plot(distMaha,distE,marker='o',color='pink')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico dist. Euclidiana por Mahalanobis (matriz covariância = identidade)")
plt.xlabel("Mahalanobis")
plt.ylabel("dist. Euclidiana")
plt.plot(distMahaI,distE,marker='o',color='gray')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico Beta0 por distância Euclidiana (x,y)")
plt.xlabel("Dist. Euclidiana")
plt.ylabel("Beta0")
plt.plot(distE,beta0,marker='o',color="orange")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico beta1 por distância Euclidiana (x,y)")
plt.xlabel("Dist. Euclidiana")
plt.ylabel("Beta1")
plt.plot(distE, beta1, marker='o',color="green")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico beta0 por distância Mahalanobis (x,y)")
plt.xlabel("Mahalanobis")
plt.ylabel("Beta0")
plt.plot(distMaha,beta0,marker='o',color="red")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico beta1 por distância Mahalanobis (x,y)")
plt.xlabel("Mahalanobis")
plt.ylabel("Beta1")
plt.plot(distMaha, beta1, marker='o', color='blue')
plt.grid(True)
plt.show()

#caso de duas variáveis explicativas

beta0 = []
beta1 = []
beta2 = []
distE = []
distMaha = []

for i in n:
    x1 = np.random.normal(20,2,i)
    x2 = np.random.normal(15,2,i)
    y = np.random.normal(10,2,i)

    dados = []
    dados.append(x1)
    dados.append(x2)
    distE.append(euclidean(x1,x2))
    covm = np.cov(dados,rowvar=False)
    incov = np.linalg.pinv(covm)
    distMaha.append(mahalanobis(x1,x2,incov))

    novox= np.column_stack((x1, x2))
    novox = sm.add_constant(novox)
    modelo = sm.OLS(y,novox).fit()

    beta0.append(modelo.params[0])
    beta1.append(modelo.params[1])
    beta2.append(modelo.params[2])

plt.figure(figsize=(8,6))
plt.title("Gráfico beta0 por distância euclidiana (x1,x2)")
plt.plot(distE,beta0,marker='o',color='purple')
plt.xlabel("Dist. Euclidiana")
plt.ylabel("Beta0")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico beta1 por distância euclidiana (x1,x2)")
plt.plot(distE,beta1,marker='o',color='brown')
plt.xlabel("Dist. Euclidiana")
plt.ylabel("Beta1")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico beta2 por distância euclidiana (x1,x2)")
plt.plot(distE,beta2,marker='o',color='turquoise')
plt.xlabel("Dist. Euclidiana")
plt.ylabel("Beta2")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico beta0 por distância Mahalanobis (x1,x2)")
plt.plot(distMaha,beta0,marker='o',color='purple')
plt.xlabel("Dist. Mahalanobis")
plt.ylabel("Beta0")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico beta1 por distância Mahalanobis (x1,x2)")
plt.plot(distMaha,beta1,marker='o',color='brown')
plt.xlabel("Dist. Mahalanobis")
plt.ylabel("Beta1")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.title("Gráfico beta2 por distância Mahalanobis (x1,x2)")
plt.plot(distMaha,beta2,marker='o',color='turquoise')
plt.xlabel("Dist. Euclidiana")
plt.ylabel("Beta2")
plt.grid(True)
plt.show()




