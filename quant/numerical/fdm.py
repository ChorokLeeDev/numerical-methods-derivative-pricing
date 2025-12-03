#%%
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.interpolate import interp1d

def exfdm_vanilla_option(s0, k, r, q, t, vol, optionType, maxS, N, M):
    ds = maxS / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds
    a = dt*(vol*s[1:-1])**2 / (2*ds**2)
    b = dt*(r-q)*s[1:-1] / (2*ds)
    d, m, u = a-b, -2*a-dt*r, a+b

    v = np.maximum(callOrPut*(s-k), 0)

    for j in range(M-1,-1,-1):
        temp = d * v[:-2] + (1 + m) * v[1:-1] + u * v[2:]
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = np.maximum(callOrPut*(maxS - k * np.exp(-r * (M - j) * dt)), 0)
        v[1:-1] = temp
    f = interp1d(s,v)
    return pd.DataFrame({"S":s,"V":v}), f(s0)


def fdm_vanilla_option(s0, k, r, q, t, vol, optionType, maxS, N, M, theta=1):
    ds = maxS / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds

    a = dt*(vol*s[1:-1])**2 / (2*ds**2)
    b = dt*(r-q)*s[1:-1] / (2*ds)
    d, m, u = a-b, -2*a-dt*r, a+b

    A = np.diag(d[1:],-1) + np.diag(m) + np.diag(u[:-1],1)
    B = np.zeros((N-1,2))
    B[0,0], B[-1,1] = d[0], u[-1]

    Am = np.identity(N-1) - theta*A
    Ap = np.identity(N-1) + (1-theta)*A
    ab = np.zeros((3, N-1))
    ab[0,1:] = np.diag(Am,1)
    ab[1] = np.diag(Am)
    ab[2,:-1] = np.diag(Am,-1)

    v = np.maximum(callOrPut*(s-k), 0)
    for j in range(M-1,-1,-1):
        #temp = Ap @ v[1:-1] + theta*B @ v[[0,-1]]
        temp = (1-theta)*d * v[:-2] + (1 + (1-theta)*m) * v[1:-1] + (1-theta)*u * v[2:]
        temp[0] += theta*d[0]*v[0]
        temp[-1] += theta*u[-1]*v[-1]
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = np.maximum(callOrPut*(maxS - k * np.exp(-r * (M - j) * dt)), 0)
        temp += (1-theta)*B @ v[[0,-1]]
        v[1:-1] = solve_banded((1,1), ab, temp)
        if j==1:
            f = interp1d(s,v)
            p1 = f(s0)

    f = interp1d(s,v)
    price = f(s0)
    delta = (f(1.01*s0) - f(0.99*s0)) / 2
    gamma = (f(1.01*s0) -2*price + f(0.99*s0)) / 2
    theta_1day = (p1-price) / dt / 365.0
    return pd.DataFrame({"S":s,"V":v}), f(s0), delta, gamma, theta_1day



def fdm_barrier_option(s0, k, r, q, t, vol, optionType, b, N, M, theta=1): #up-out barrier option
    ds = b / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds

    a = dt*(vol*s[1:-1])**2 / (2*ds**2)
    b = dt*(r-q)*s[1:-1] / (2*ds)
    d, m, u = a-b, -2*a-dt*r, a+b

    A = np.diag(d[1:],-1) + np.diag(m) + np.diag(u[:-1],1)
    B = np.zeros((N-1,2))
    B[0,0], B[-1,1] = d[0], u[-1]

    Am = np.identity(N-1) - theta*A
    Ap = np.identity(N-1) + (1-theta)*A
    ab = np.zeros((3, N-1))
    ab[0,1:] = np.diag(Am,1)
    ab[1] = np.diag(Am)
    ab[2,:-1] = np.diag(Am,-1)

    v = np.maximum(callOrPut*(s-k), 0)
    for j in range(M-1,-1,-1):
        #temp = Ap @ v[1:-1] + theta*B @ v[[0,-1]]
        temp = (1-theta)*d * v[:-2] + (1 + (1-theta)*m) * v[1:-1] + (1-theta)*u * v[2:]
        temp[0] += theta*d[0]*v[0]
        temp[-1] += theta*u[-1]*v[-1]
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = 0 #Knock-out
        temp += (1-theta)*B @ v[[0,-1]]
        v[1:-1] = solve_banded((1,1), ab, temp)
        if j==1:
            f = interp1d(s,v)
            p1 = f(s0)

    f = interp1d(s,v)
    price = f(s0)
    delta = (f(1.01*s0) - f(0.99*s0)) / 2
    gamma = (f(1.01*s0) -2*price + f(0.99*s0))
    theta_1day = (p1-price) / dt / 365.0
    return pd.DataFrame({"S":s,"V":v}), f(s0), delta, gamma, theta_1day


if __name__=="__main__":
    s = 100
    k = 100
    r = 0.03
    q = 0.00
    t = 1
    sigma = 0.25
    optionType = 'call'
    maxS, n, m = s*4, 1600, 2000
    print("="*50)
    print("Vanilla Option")
    print("="*50)
    '''
    v, ex_price = exfdm_vanilla_option(s, k, r, q, t, sigma, optionType,
                                    maxS, n, m)
    print(f"EX-FDM Price = {ex_price:0.6f}")

    v, ex_price, delta, gamma, theta = fdm_vanilla_option(s, k, r, q, t, sigma, optionType,
                                    maxS, n, m, 0)
    print(f"EX-FDM Price = {ex_price:0.6f}")
    '''

    v, im_price, delta, gamma, theta = fdm_vanilla_option(s, k, r, q, t, sigma, optionType,
                                    maxS, n, m)
    print(f"IM-FDM Price = {im_price:0.6f}, Delta = {delta:0.4f}, Gamma = {gamma:0.4f}, Theta = {theta:0.4f}")

    v, cn_price, delta, gamma, theta = fdm_vanilla_option(s, k, r, q, t, sigma, optionType,
                                    maxS, n, m, 0.5)
    print(f"CN-FDM Price = {cn_price:0.6f}, Delta = {delta:0.4f}, Gamma = {gamma:0.4f}, Theta = {theta:0.4f}")


    from src.pricing.blackscholes import bsprice
    bs_price = bsprice(s, k, r, q, t, sigma, optionType)
    print(f"Analytic Price = {bs_price:0.6f}")

    #Barrier Option Pricing
    print()
    print("="*50)
    print("Barrier Option")
    print("="*50)
    b = 150
    v, cn_price, delta, gamma, theta = fdm_barrier_option(s, k, r, q, t, sigma, optionType,
                                    b, n, m, 0.5)
    print(f"CN-FDM Price = {cn_price:0.6f}, Delta = {delta:0.4f}, Gamma = {gamma:0.4f}, Theta = {theta:0.4f}")

    # Note: ql_barrier_option module not available in this repo
    # from src.pricing.ql_barrier_option import ql_barrier_price
    # ql_price = ql_barrier_price(s, k, r, t, sigma, optionType,
    #                             b, 0, "up-out")
    # print(f"Analytic Price = {ql_price:0.6f}")
