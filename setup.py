from fipy import *
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import  make_axes_locatable
from tqdm import tqdm

## Definitions

def temp_dependency(x):
    sig = 5
    mu = 42
    return exp(-power(x - mu, 2.) / (2 * power(sig, 2.)))


def arrhenius(x,T, E_a):
    gas_const = 8.314446
    T = T + 273.15
    return x*exp(-E_a / (gas_const * T))

def water_activity(moisture_content, T, time):
    # moisture_content - moisture content in parts
    # T - temperature in Kelvin
    # time - time in days
    rate = .01727 + .02613*(.9359**time)
    pow  = .69910 + .41730*(.9434**time)
    T = T + 273.15
    return 1 - exp(-T*rate*(moisture_content**pow))

## Tests and plots

if __name__ == '__main__':
    # Water Activity against moisture content for different temperatur
    tempr_w_cont = arange(0., 90., 15.)
    times = arange(0., 56., 7.)
    numRows = int(ceil(len(times)/2))
    fig1, axs1 = plt.subplots(2, numRows)
    moistures = arange(.1, 1.1, .1)
    fig1.set_size_inches(13.0, 6.0)
    for j in range(len(times)):
        for i in range(len(tempr_w_cont)):
            a_w = water_activity(moistures, tempr_w_cont[i], times[j])
            axCol = int(j/numRows)
            axRow = int(j%numRows)
            axs1[axCol, axRow].plot(moistures, a_w, label='T= %.3f' % (tempr_w_cont[i]))
            axs1[axCol, axRow].legend()
            axs1[axCol, axRow].set_xlabel('Moisture content')
            axs1[axCol, axRow].set_ylabel('Water activity')
            axs1[axCol, axRow].set_title('%.2f days'%(times[j]))
            axs1[axCol, axRow].xaxis.set_ticks(arange(.1, 1.1, .1))
    plt.tight_layout()
    plt.savefig('Water activity vs moisture content')
    # Arrhenius curves for different activation energy values:
    Death_u, Death_v = .1, .2
    tempr_plot = arange(10., 100., 10.)
    energy_plot = arange(1., 5., 1.)
    fig, axs = plt.subplots(2, 1)
    for e_power in energy_plot:
        energy = 10. ** e_power
        rates_u = arrhenius(Death_u, tempr_plot, energy)
        rates_v = arrhenius(Death_v, tempr_plot, energy)
        axs[0].plot(tempr_plot, log(rates_u), label='E_a= %.3f' % (energy))
        axs[1].plot(tempr_plot, log(rates_v), label='E_a= %.3f' % (energy))
    axs[0].legend()
    axs[0].set_xlabel('T, C')
    axs[0].set_ylabel('log(r_d), for u')
    axs[1].legend()
    axs[1].set_xlabel('T, C')
    axs[1].set_ylabel('log(r_d), for v')
    plt.show()
    plt.savefig('Arrhenius plot')




