# SIR模型基础上增加潜伏期和二次感染
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

def SEIIR_model(y, t, beta, sigma, gamma1, gamma2):

    S, E, I, R, II = y

    dS_dt = -beta * S * I

    dE_dt = beta * S * I - sigma * E

    dI_dt = sigma * E - gamma1 * I

    dR_dt = gamma1 * I - gamma2 * R

    dII_dt = gamma2 * R

    return [dS_dt, dE_dt, dI_dt, dR_dt, dII_dt]


S0 = 0.8  # 易感染者初始比例
E0 = 0.1  # 潜伏期初始比例
I0 = 0.1  # 感染者初始比例
R0 = 0.0  # 免疫病毒者初始比例
II0 = 0.0  # 二次感染者初始比例
beta = 0.3  # 一个病人能传染的易感染者数目与此环境易感染者总人数比值
gamma1 = 0.1  # 单位时间内感染者痊愈成免疫者的概率
gamma2 = 0.05  # 免疫者再次感染成为感染者的概率
sigma = 0.2  # 患者从暴露到发病的时间（潜伏期）

t = np.linspace(0, 100, 10000)

res = scipy.integrate.odeint(SEIIR_model, [S0, E0, I0, R0, II0], t, args=(beta, sigma, gamma1, gamma2))
res = np.array(res)

plt.figure(figsize=[8, 6])
plt.plot(t, res[:, 0], label='S(t)')
plt.plot(t, res[:, 1], label='E(t)')
plt.plot(t, res[:, 2], label='I(t)')
plt.plot(t, res[:, 3], label='R(t)')
plt.plot(t, res[:, 4], label='II(t)')
plt.legend()
plt.grid()
plt.xlabel('time')
plt.ylabel('proportions')
plt.title('SEIIR model simulation')
plt.show()
