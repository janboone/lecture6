import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.title("Predator–Prey Dynamics and a Second Order Differential Equation")

st.markdown(r"""
The **predator–prey model** is one of the most famous examples of a dynamic
system. It was introduced by **Lotka and Volterra** in the 1920s to describe
how two interacting populations evolve over time.

The basic idea is simple:

- When there are many prey (for example rabbits), predators (for example foxes)
  have plenty of food and their population grows.
- When there are many predators, the prey population declines because more
  prey are eaten.
- When prey become scarce, predators start to starve and their population
  declines again.

This interaction often generates **cycles**: prey rise first, then predators
follow, after which prey fall and predators eventually fall as well.

Models like this appear in many fields beyond ecology. Similar dynamics arise
in:

- economics (for example interacting markets or business cycles),
- epidemiology (spread of diseases),
- chemical reactions,
- and strategic interactions in game theory.

Studying this model helps illustrate several important ideas:

- how to formulate **dynamic models using differential equations**,
- how interacting variables generate **feedback and cycles**,
- and how to **solve systems of differential equations numerically in Python**.

The classic **Lotka–Volterra predator–prey model** is

$$
\dot{x} = ax - bxy
$$

$$
\dot{y} = -cy + dxy
$$

where

- $x(t)$ = prey population
- $y(t)$ = predator population

The parameters have intuitive interpretations:

- $a$ : natural growth rate of prey
- $b$ : intensity of predation
- $c$ : natural death rate of predators
- $d$ : efficiency with which predators convert prey into new predators

""")

st.sidebar.header("Model interpretation")

model_type = st.sidebar.radio(
    "Choose interpretation",
    ["Predator–prey (ecology)", "Business cycle (Goodwin model)"]
)

st.sidebar.header("Parameters")

a = st.sidebar.slider("a",0.1,3.0,1.0)
b = st.sidebar.slider("b",0.1,2.0,0.5)
c = st.sidebar.slider("c",0.1,3.0,1.5)
d = st.sidebar.slider("d",0.1,2.0,0.75)

st.sidebar.header("Initial values")

x0 = st.sidebar.slider("Initial x",0.5,10.0,4.0)
y0 = st.sidebar.slider("Initial y",0.5,10.0,2.0)


def lotka_volterra(t, z):
    x, y = z

    dx = a*x - b*x*y
    dy = -c*y + d*x*y

    return [dx, dy]


T = 40
sol = solve_ivp(lotka_volterra,(0,T),[x0,y0],t_eval=np.linspace(0,T,500))

x = sol.y[0]
y = sol.y[1]

if model_type == "Predator–prey (ecology)":
    x_label = "prey"
    y_label = "predators"
else:
    x_label = "employment rate"
    y_label = "wage share"

fig1,ax1 = plt.subplots()
ax1.plot(sol.t,x,label=f"{x_label} x(t)")
ax1.plot(sol.t,y,label=f"{y_label} y(t)")
ax1.set_xlabel("time")
ax1.legend()

fig2,ax2 = plt.subplots()
ax2.plot(x,y)
ax2.set_xlabel(x_label)
ax2.set_ylabel(y_label)
ax2.set_title("Phase diagram")

col1,col2 = st.columns(2)

with col1:
    st.pyplot(fig1)

with col2:
    st.pyplot(fig2)

st.markdown(r"""
## An economic interpretation: business cycles (Goodwin model)

The same mathematical structure can also describe **economic cycles**.  
A famous example is the **Goodwin growth cycle model** (1967), which applies
Lotka–Volterra dynamics to macroeconomics.

Instead of animals, the interacting variables are:

- $x(t)$ : the **employment rate** in the economy
- $y(t)$ : the **wage share** of income (the fraction of output going to workers)

The intuition is similar to predator–prey interactions.

**Step 1 — High employment increases wages**  
When employment is high, workers have stronger bargaining power. Firms must
raise wages to attract or keep workers. As a result, the **wage share rises**.

**Step 2 — High wages reduce profits and hiring**  
When wages take a larger share of output, firms' profits fall. Firms invest and
hire less, so **employment begins to decline**.

**Step 3 — Low employment weakens worker bargaining power**  
When unemployment rises, workers compete for jobs and wage growth slows. The
**wage share falls**.

**Step 4 — Lower wages restore profitability**  
With lower wages, firms become more profitable again. Investment and hiring
increase, and **employment rises again**.

This feedback loop generates **endogenous business cycles**: employment rises
first, wages follow, then employment falls, followed by falling wages.

Mathematically, the Goodwin model can be written in a form that closely
resembles the Lotka–Volterra system:

$$
\dot{x} = ax - bxy
$$

$$
\dot{y} = -cy + dxy
$$

where now

- $x$ = employment rate
- $y$ = wage share

In this interpretation:

- $a$ is the **baseline growth rate of employment** when wage pressure is low
- $b$ captures how a higher **wage share** reduces employment growth (because higher wages reduce profits and hiring incentives)
- $c$ captures the **baseline downward pressure on the wage share** (for example due to unemployment, labour market competition, or institutional factors)
- $d$ captures how tight labour markets push wages up

To see the intuition more clearly, rewrite both equations as

$$
\dot{x} = x(a - by)
$$

$$
\dot{y} = y(-c + dx)
$$

The wage share therefore **falls when employment is low** (because the term
$-c$ dominates) and **rises when employment becomes sufficiently high** so
that $dx > c$. This captures the idea of a **Phillips‑curve type mechanism**:
tight labour markets increase workers' bargaining power and push wages up.

The **same mathematics** therefore describes both

- ecological predator–prey cycles, and
- macroeconomic business cycles.

This is one reason why dynamic systems are so useful: once we understand the
structure of the equations, we can apply them in many different fields.

---

## Exercises

Use the **sliders in the sidebar** to explore how the dynamics change.

**Question 1**  
Try to find parameter values where the **cycles become very large** (large swings
in the two variables). Which parameters seem most important in determining how
big the fluctuations are?

**Question 2**  
Find a combination of parameters and initial values where the variables change
**very slowly** over time and the cycle becomes long. Which parameter changes
slow down the dynamics of the system?

Explain the economic intuition behind the parameter changes you used.
""")



