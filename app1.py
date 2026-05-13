import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

st.title("Ramsey Growth Model")

st.markdown(r"""
This interactive app illustrates the **Ramsey optimal growth model**.

Instead of assuming an exogenous saving rate $s$ as in the Solow model, here we endogenize the agent's saving decision.

A representative household chooses consumption to maximize

$$
\max \int_0^{\infty} e^{-\rho t} u(c(t)) \, dt
$$

subject to capital accumulation

$$
\dot{k} = f(k) - \delta k - c
$$

We can solve this problem using the **Euler equation** for dynamic optimization. Consider

$$
\mathcal{L}=e^{-\rho t}\left(u(c(t)) - \lambda(t)(\dot{k} - f(k) + \delta k + c)\right)
$$

The first‑order conditions are as follows.

Consumption:

$$
\frac{\partial \mathcal{L}}{\partial c}=e^{-\rho t}(u'(c)-\lambda)=0
$$

Capital (Euler equation):

$$
\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{k}}\right)-\frac{\partial \mathcal{L}}{\partial k}=0
$$

which implies

$$
\dot{\lambda}=(\rho+\delta-f'(k))\lambda
$$

Using the condition $\lambda=u'(c)$ and differentiating gives --what is called-- the **Euler equation for consumption**

$$
\frac{u''(c)}{u'(c)} \dot{c} = -(f'(k) - \delta - \rho)
$$

With the functional forms

$$
u(c) = c^{0.7}, \qquad f(k) = k^{0.5}
$$

The dynamic system becomes

$$
\dot{k} = k^{0.5} - \delta k - c
$$

$$
\dot{c} = \frac{c}{0.3}\left(0.5 k^{-0.5} - \delta - \rho\right)
$$
""")

st.sidebar.header("Parameters")

# Restrict parameters to avoid explosive dynamics
rho = st.sidebar.slider("Discount rate ρ",0.01,0.06,0.03,0.005)
delta = st.sidebar.slider("Depreciation δ",0.02,0.08,0.05,0.005)

st.sidebar.header("Initial conditions")

k0 = st.sidebar.slider("Initial capital k0",5.0,50.0,20.0)
c0 = st.sidebar.slider("Initial consumption c0",0.5,4.0,2.0)

# steady state
k_star = (0.5/(delta+rho))**2
c_star = np.sqrt(k_star) - delta*k_star


st.header("Steady state outcome")

st.write("We define the steady state as the values $(k^*,c^*)$ where $\dot{k} = \dot{c} = 0$")
st.write("With the functional forms above and the selected parameter values we find the following.")
st.write("Steady state capital k* =",round(k_star,3))
st.write("Steady state consumption c* =",round(c_star,3))


def ramsey_rhs(t,y):
    k,c = y

    # keep variables in numerically safe region
    k = max(k,1e-8)
    c = max(c,1e-8)

    dk = np.sqrt(k) - delta*k - c
    dc = (c/0.3)*(0.5/np.sqrt(k) - delta - rho)

    return [dk,dc]

# solve dynamics
# Stop integration if capital becomes negative or extremely large
def stop_event(t,y):
    k = y[0]
    return min(k-1e-4,100-k)
stop_event.terminal = True

sol = solve_ivp(ramsey_rhs,(0,80),[k0,c0],t_eval=np.linspace(0,80,400),events=stop_event,max_step=0.5)

k = sol.y[0]
c = sol.y[1]

# time paths
fig1,ax1 = plt.subplots()
line_k, = ax1.plot(sol.t,k,label="capital k(t)")
line_c, = ax1.plot(sol.t,c,label="consumption c(t)")

# steady states with matching colors
ax1.axhline(k_star,linestyle="--",color=line_k.get_color(),label="k*")
ax1.axhline(c_star,linestyle="--",color=line_c.get_color(),label="c*")

ax1.set_xlabel("time")
ax1.legend()

# phase diagram
k_max = 100
kgrid = np.linspace(0.1,k_max,120)
cgrid = np.linspace(0.1,6,80)
K,C = np.meshgrid(kgrid,cgrid)

dK = K**0.5 - delta*K - C
dC = (C/0.3)*(0.5*K**-0.5 - delta - rho)

fig2,ax2 = plt.subplots()
ax2.streamplot(K,C,dK,dC,color="gray",density=1)

c_kdot0 = kgrid**0.5 - delta*kgrid
ax2.plot(kgrid,c_kdot0,label="k'=0")
ax2.axvline(k_star,linestyle="--",label="c'=0")

ax2.plot(k,c,color="red",label="trajectory")
ax2.scatter([k_star],[c_star])

ax2.set_xlim(0,k_max)
ax2.set_ylim(0,6)
ax2.set_xlabel("capital k")
ax2.set_ylabel("consumption c")
ax2.legend()

# show first two figures side-by-side
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)

# -------- Saddle path calculation --------
st.header("Saddle path consumption for given k₀")

# function used for shooting method
def terminal_k(c0_guess):
    sol_tmp = solve_ivp(ramsey_rhs,(0,120),[k0,c0_guess],max_step=0.5)
    return sol_tmp.y[0,-1] - k_star

try:
    root = root_scalar(terminal_k,bracket=[0.2,5],method="brentq")
    c0_saddle = root.root

    st.write("For given initial capital stock $k_0$, there is only one value $c_0$ that sets the economy on the path to the steady state.")
    st.write(f"Consumption on the saddle path for $k_0$ = {k0:.2f} is $c_0$ ≈ {c0_saddle:.3f}")

    sol_saddle = solve_ivp(ramsey_rhs,(0,80),[k0,c0_saddle],t_eval=np.linspace(0,80,400),max_step=0.5)

    # ---- add saddle path to phase diagram ----
    ax2.plot(sol_saddle.y[0],sol_saddle.y[1],"--",color="red",label="saddle path")
    ax2.legend()

    # ---- time dynamics on saddle path ----
    fig3,ax3 = plt.subplots()
    line_k_s, = ax3.plot(sol_saddle.t,sol_saddle.y[0],label="capital k(t)")
    line_c_s, = ax3.plot(sol_saddle.t,sol_saddle.y[1],label="consumption c(t)")

    ax3.axhline(k_star,linestyle="--",color=line_k_s.get_color(),label="k*")
    ax3.axhline(c_star,linestyle="--",color=line_c_s.get_color(),label="c*")

    ax3.set_xlabel("time")
    ax3.set_title("Dynamics on the saddle path")
    ax3.legend()

    # show saddle dynamics (left) and phase diagram (right)
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(fig3)
    with col4:
        st.pyplot(fig2)

except ValueError:
    st.warning("Could not find a saddle-path value for c₀ with the current parameters.")


# -------- Exercises for students --------
st.header("Exercises")

st.markdown("""
**Question 1.** Use the sliders for the parameters $\\rho$ and $\delta$ to find a situation where the steady‑state capital level $k^*$ is **approximately equal to 25**.  

- What values of $\\rho$ and $\delta$ achieve this?  
- What is the corresponding steady‑state consumption $c^*$?

**Question 2.** Choose parameters $\\rho = 0.03$ and $\delta = 0.05$. Now change the initial capital level $k_0$.

- Find one value of $k_0$ for which the economy must **initially reduce consumption** (consumption falls before converging to the steady state).
- Find one value of $k_0$ for which the economy must **initially increase consumption**.

Explain your answer using the **phase diagram** and the **saddle path**.
""")
