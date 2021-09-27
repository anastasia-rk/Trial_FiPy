from numpy import *
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
## Definitions

def temp_dependency(x):
    sig = 5
    mu = 42
    return exp(-power(x - mu, 2.) / (2 * power(sig, 2.)))


def arrhenius(x,T, E_a):
    # commonly used model for temperature-dependent reaction rates
    gas_const = 8.314446
    T = T + 273.15
    return x*exp(-E_a / (gas_const * T))

def sharpe_demichele(T, enthalpies, entropies):
    # activation driven by a single rate-limiting enzyme-catalysed reaction
    # the rate of this reaction is Arrhenius, with additional thermodynamic terms
    # quantifying rate-modifying effect of low and high temperatures
    gas_const = 8.314446
    T = T + 273.15
    return T*exp((entropies[0] - enthalpies[0]/T)/gas_const)/(1 + exp((entropies[1] - enthalpies[1]/T)/gas_const) + exp((entropies[2] - enthalpies[2]/T)/gas_const))

def schoolfield(T, enthalpies, T_high, T_low, phi):
    # activation driven by a single rate-limiting enzyme-catalysed reaction
    # the rate of this reaction is Arrhenius, with additional thermodynamic terms
    # quantifying rate-modifying effect of low and high temperatures
    # enthalpies - heat of activation
    # phi - a constant encompassing the entropy of activation of the rate-controlling enzyme
    gas_const = 8.314446
    T = T + 273.15
    rho = 298*exp(phi - enthalpies[0]/gas_const)
    return (rho*T*exp(enthalpies[0]*(1/298 - 1/T)/gas_const)/298)/(1 + exp(enthalpies[1]*(1/T_low - 1/T)/gas_const) + exp(enthalpies[2]*(1/T_high - 1/T)/gas_const))

def cardinal_model(x, x_opt, x_min, x_max):
    # model originally proposed for temerpature dependent growth rate in Rosso et.al. 1995
    # used as gamma parameter for growth inhibition in response to PH
    gamma = (x - x_min)*(x - x_max)/((x - x_min)*(x - x_max) - (x - x_opt)**2)
    indeces = [i for i in range(len(x)) if x[i] < x_min or x[i] > x_max]
    gamma[indeces] = 0
    return gamma

def ratkowsky_model(x, x_min, x_max, b, c):
    # Ratkowsky model only includes min temperature and defines the square root of the growth rate
    return (b*(x - x_min)*(1 - exp(c*(x - x_max))))**2

def mcmeekin_model(a_w, a_w_min):
    return a_w - a_w_min

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
    # Water Activity against moisture content for different temperature
    tempr_w_cont = arange(0., 90., 15.)
    times = arange(0., 56., 7.)
    numRows = int(ceil(len(times)/2))
    fig1, axs1 = plt.subplots(2, numRows)
    moistures = arange(.1, 1.1, .1)
    aw_min_ecol = .95 * ones(len(moistures))
    aw_min_bacil_c = .92 * ones(len(moistures))
    aw_min_clostr_b = .935 * ones(len(moistures))
    aw_min_fe = .97 * ones(len(moistures))
    fig1.set_size_inches(13.0, 6.0)
    for j in range(len(times)):
        for i in range(len(tempr_w_cont)):
            a_w = water_activity(moistures, tempr_w_cont[i], times[j])
            axCol = int(j/numRows)
            axRow = int(j%numRows)
            axs1[axCol, axRow].plot(moistures, a_w, label='T= %.3f' % (tempr_w_cont[i]))
            axs1[axCol, axRow].plot(moistures, aw_min_ecol)
            axs1[axCol, axRow].plot(moistures, aw_min_bacil_c)
            axs1[axCol, axRow].plot(moistures, aw_min_clostr_b)
            axs1[axCol, axRow].plot(moistures, aw_min_fe)
            axs1[axCol, axRow].legend()
            axs1[axCol, axRow].set_xlabel('Moisture content')
            axs1[axCol, axRow].set_ylabel('Water activity, $a_w$')
            axs1[axCol, axRow].set_title('%.2f days'%(times[j]))
            axs1[axCol, axRow].xaxis.set_ticks(arange(.1, 1.1, .1))
    plt.tight_layout()
    plt.savefig('Water activity vs moisture content')

    # pH inactivation curve - arbitrary values
    phs = arange(4., 10.5, .1)
    temperatures = arange(10., 60., .1)
    rates = cardinal_model(phs, 7.5, 5.1, 9.5)
    rates_ecol = cardinal_model(phs, 7, 4, 10)
    rates_fe   = cardinal_model(phs, 7, 5, 9.6)
    rates_bacil_c = cardinal_model(phs, 7, 4.9, 9.3)
    rates_clostr_b = cardinal_model(phs, 7, 4.6, 9)
    t_rates_ecol = cardinal_model(temperatures, 40.3, 5.6, 47.3)
    t_rates_fe   = cardinal_model(temperatures, 42, .1, 53.5)
    t_rates_bacil_c = cardinal_model(temperatures, 40.1, 4.1, 50.0)
    t_rates_clostr_b = cardinal_model(temperatures, 39.3, 11.0, 45.8)
    fig2, axs2 = plt.subplots(1, 2)
    fig2.set_size_inches(13.0, 6.0)
    axs2[0].plot(phs, rates_ecol,  label='$E.coli$')
    axs2[0].plot(phs, rates_bacil_c,  label='$B.cereus$')
    axs2[0].plot(phs, rates_clostr_b,  label='$C.botulinum$')
    axs2[0].plot(phs, rates_fe,  label='$E.faecium$')
    axs2[0].legend()
    axs2[0].set_xlabel('pH')
    axs2[0].set_ylabel('$\gamma(\mathrm{pH})$')
    axs2[0].set_title ('Growth rate factor vs pH')
    axs2[1].plot(temperatures, t_rates_ecol,  label='$E.coli$')
    axs2[1].plot(temperatures, t_rates_bacil_c,  label='$B.cereus$')
    axs2[1].plot(temperatures, t_rates_clostr_b,  label='$C.botulinum$')
    axs2[1].plot(temperatures, t_rates_fe,  label='$E.faecium$')
    axs2[1].legend()
    axs2[1].set_xlabel('T, $^{\circ}$C')
    axs2[1].set_ylabel('$\gamma(\mathrm{T})$')
    axs2[1].set_title ('Growth rate factor vs T')
    plt.tight_layout()
    plt.savefig('Inhibition of growth rates')

    # Arrhenius curves for different activation energy values:
    Death_u, Death_v = .1, .2
    tempr_plot = arange(10., 100., 10.)
    energy_plot = arange(1., 5., 1.)
    fig, axs = plt.subplots(2, 1)
    for e_power in energy_plot:
        energy = 10. ** e_power
        rates_u = arrhenius(Death_u, tempr_plot, energy)
        rates_v = arrhenius(Death_v, tempr_plot, energy)
        axs[0].plot(tempr_plot, log(rates_u), label='$E_a=$ %.3f' % (energy))
        axs[1].plot(tempr_plot, log(rates_v), label='$E_a=$ %.3f' % (energy))
    axs[0].legend()
    axs[0].set_xlabel('T, C')
    axs[0].set_ylabel('$\log(r_d)$, for u')
    axs[1].legend()
    axs[1].set_xlabel('T, C')
    axs[1].set_ylabel('$\log(r_d)$, for v')
    plt.show()
    plt.savefig('Arrhenius plot for death rates')
#   Schoolfield growthrates
    temprs = arange(10., 43., 1.)
    a_ws = arange(0.96, 1.01, .01)
    growths_ec = empty([len(a_ws), len(temprs)])
    # Data from ComBase - depending on a_w above activation and T - at PH=7
    # growths_ec[0, :] = array([.009, .011, .013, .016, .020, .024, .029, \
    #                           .034, .040, ])
    # growths_ec[1, :] = array([])
    # growths_ec[2, :] = array([])
    # growths_ec[3, :] = array([])
    growths_ec[4, :] = array([.027, .035, .044, .056, .069, .086, .105, \
                              .126, .151, .179, .210, .244, .281, .320, \
                              .361, .403, .445, .487, .527, .566, .601, \
                              .631, .657, .677, .691, .698, .698, .691, \
                              .678, .658, .632, .602, .567])
    # growths_ec[5, :] = array([])
    temprs_kelvin = 1/(temprs + 273.15)
    fig = plt.figure()
    ax = fig.add_subplot( 111 )
    ax.plot((temprs_kelvin),(growths_ec[4, :]))
    ax_top = ax.secondary_xaxis('top')
    ax.set_xlabel('1/T, 1/Kelvin')
    ax.set_ylabel('growth rate, $h^{-1}$')
    plt.savefig('Arrhenius plor growth rate')






