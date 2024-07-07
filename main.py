import numpy as np
import multiprocessing as mp
import random
import os
import time
import matplotlib.pyplot as plt
from functools import partial
import time
random.seed(time.localtime().tm_sec)
n_cpu = mp.cpu_count()
data = np.loadtxt("test2_600.txt", unpack=True)
tempo = data[0]
periodo = data[1]
w = 2.05 * 10 ** (-2)
g = 9.81
err_w = 0.005 * 10 ** (-2)
l = 1.15
err_l = 0.001
d = 1.20
err_d = 0.001
transit_time = data[2]
err_trans_time = np.full(shape=len(transit_time), fill_value=(1 * 10 ** (-4)), dtype=float)
err_periodo = np.full(shape=len(periodo), fill_value=1 * 10 ** (-4), dtype=float)

def init_process(chisq_shared_, l_shared_, lock_):
    global chisq_shared, l_shared, lock
    chisq_shared = chisq_shared_
    l_shared = l_shared_
    lock = lock_

def _chisq(l, modello_func, _modello_1):
    k = (w ** 2 * l) / (2 * (transit_time ** 2) * (d ** 2) * g)
    sigma_k = np.sqrt(2 * (err_w / w) ** 2 + (err_l / l) ** 2 + 2 * (err_trans_time / transit_time) ** 2 + 2 * (err_d / d) ** 2) * k
    theta_0 = np.arccos(1 - k)
    err_theta_0 = (1 / np.sqrt(1 - (1 - k) ** 2)) * sigma_k

    local_chisq = float('inf')
    if _modello_1:
        local_chisq = min((((periodo - modello_func(theta_0, l)) ** 2 / (err_periodo** 2))).sum(), local_chisq)
    else:
        local_chisq = min((((periodo - modello_func(theta_0, l)) ** 2 / (err_periodo** 2))).sum(), local_chisq)
    
    with lock: # Lock fa in modo che non mi si corrompa la memoria se più processi tentano di accedere
        if local_chisq < chisq_shared.value:
            chisq_shared.value = local_chisq
            l_shared.value = l
    return local_chisq, l

def modello_1(x, l):
    return 2 * np.pi * np.sqrt(l/g) * (1 + (1. / 16.) * x ** 2.0)

def modello_2(x, l):
    return 2 * np.pi * np.sqrt(l/g) * (1 + (1. / 16.) * x ** 2.0 + (11. / 3072.) * x ** 4.0)

if __name__ == "__main__":
    linspace = np.linspace(0.90, 1.80, num=100000)
    print(linspace)
    # Inizializzazione variabile comune e lock per l'accesso 
    chisq_shared = mp.Value('d', float('inf'))
    l_shared = mp.Value('d', 0.0)
    lock = mp.Lock()

    # Passaggi di "parametri" tramite partial
    chisq_model1 = partial(_chisq, modello_func=modello_1, _modello_1=True)
    # Inizio misurazione del tempo
    t0 = time.time()
    # Pool del modello_1
    pool = mp.Pool(processes=n_cpu, initializer=init_process, initargs=(chisq_shared, l_shared, lock))
    result1 = pool.map(chisq_model1, linspace)
    t1 = time.time()
    print(f"Tempo di esecuzione primo modello: {t1 - t0}")
    # Chiusura pool del modello 1
    pool.close()
    pool.join()
    # 
    # Codice per il primo grafico
    # 
    # k = (w ** 2 * l_shared.value) / (2 * (transit_time ** 2) * (d ** 2) * g)
    # sigma_k = np.sqrt(2 * (err_w / w) ** 2 + (err_l / l_shared.value) ** 2 + 2 * (err_trans_time / transit_time) ** 2 + 2 * (err_d / d) ** 2) * k
    # theta_0 = np.arccos(1 - k)
    # err_theta_0 = (1 / np.sqrt(1 - (1 - k) ** 2)) * sigma_k
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    # dof = len(periodo)
    # Plot principale: dati e modello di best-fit
    # ax1.errorbar(theta_0, periodo, err_periodo, fmt="o", label="Dati")
    # xgrid = np.linspace(0.0, 0.25, 100)
    # ax1.plot(xgrid, modello_2(xgrid, l_shared.value), label=f'Best-fit model (χ² ridotto: {chisq_shared.value:.2f})', color="orange")
    # ax1.set_ylabel("T [s]")
    # ax1.grid(color="lightgray", ls="dashed")
    # ax1.legend()
    # res = periodo - modello_1(theta_0, l_shared.value)
    # Titolo del grafico principale
    # ax1.set_title(f"Fit per la stima dell'indice con χ²: {chisq_shared.value:.2f}, Gradi di Libertà: {dof}")

    # Plot dei residui
    # ax2.errorbar(theta_0, res, err_periodo, fmt="o")
    # ax2.plot(xgrid, np.zeros_like(xgrid), color='orange')
    # ax2.set_xlabel("theta_0 [rad]")
    # ax2.set_ylabel("Residui [s]")
    # ax2.grid(color="lightgray", ls="dashed")

    # Finalizzazione del grafico
    # plt.xlim(0.0, 0.25)
    # fig.align_ylabels((ax1, ax2))
    # plt.savefig("Fit_e_residui_modello_1.pdf")
    # plt.show()
    print("Minimo valore del chi2 col primo modello:", chisq_shared.value)
    print("Valore di l che minimizza il chi2 col primo modello:", l_shared.value)

    # Reset dei valori condivisi fra i processi
    chisq_shared.value = float('inf')
    l_shared.value = 0.0

    # Passaggio sempre dei "parametri"
    chisq_model2 = partial(_chisq, modello_func=modello_2, _modello_1=False)
    t0 = time.time()
    # Pool for modello_2
    pool = mp.Pool(processes=n_cpu, initializer=init_process, initargs=(chisq_shared, l_shared, lock))
    result2 = pool.map(chisq_model2, linspace)
    t1 = time.time()
    print(f"Tempo di esecuzione secondo modello: {t1-t0}")
    pool.close()
    pool.join()

    print("Minimo valore del chi2 col secondo modello:", chisq_shared.value)
    print("Valore di l che minimizza il chi2", l_shared.value)

    # Codice per il secondo grafico
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    # dof = len(periodo)
    # Plot principale: dati e modello di best-fit
    # ax1.errorbar(theta_0, periodo, err_periodo, fmt="o", label="Dati")
    # xgrid = np.linspace(0.0, 0.25, 100)
    # ax1.plot(xgrid, modello_2(xgrid, l_shared.value), label=f'Best-fit model (χ² ridotto: {chisq_shared.value:.2f})', color="orange")
    # ax1.set_ylabel("T [s]")
    # ax1.grid(color="lightgray", ls="dashed")
    # ax1.legend()
    # res = periodo - modello_2(theta_0, l_shared.value)
    # Titolo del grafico principale
    # ax1.set_title(f"Fit per la stima dell'indice con χ²: {chisq_shared.value:.2f}, Gradi di Libertà: {dof}")

    # Plot dei residui
    # ax2.errorbar(theta_0, res, err_periodo, fmt="o")
    # ax2.plot(xgrid, np.zeros_like(xgrid), color='orange')
    # ax2.set_xlabel("theta_0 [rad]")
    # ax2.set_ylabel("Residui [s]")
    # ax2.grid(color="lightgray", ls="dashed")

    # Finalizzazione del grafico
    # plt.xlim(0.0, 0.25)
    # fig.align_ylabels((ax1, ax2))
    # plt.savefig("Fit_e_residui_modello_2.pdf", dpi=500)
    # plt.show()



