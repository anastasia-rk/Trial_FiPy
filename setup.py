from numpy import *
## Plot settings
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib as tkz
# matplotlib.use('TkAgg')
# plt.style.use("ggplot")
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

def schoolfield(T, enthalpies, T_high, T_low, rho):
    # activation driven by a single rate-limiting enzyme-catalysed reaction
    # the rate of this reaction is Arrhenius, with additional thermodynamic terms
    # quantifying rate-modifying effect of low and high temperatures
    # enthalpies - heat of activation
    # phi - a constant encompassing the entropy of activation of the rate-controlling enzyme
    # gas_const = 8.314446
    gas_const = 1.987
    T = T + 273.15
    # rho = 298*exp(phi - enthalpies[0]/gas_const)
    return (rho*T*exp(enthalpies[0]*(1/298 - 1/T)/gas_const)/298)/(1 + exp(enthalpies[1]*(1/T_low - 1/T)/gas_const) + exp(enthalpies[2]*(1/T_high - 1/T)/gas_const))

def cardinal_model(x, x_opt, x_min, x_max):
    # model originally proposed for temerpature dependent growth rate in Rosso et.al. 1995
    # used as gamma parameter for growth inhibition in response to PH
    gamma = (x - x_min)*(x - x_max)/((x - x_min)*(x - x_max) - (x - x_opt)**2)
    indeces = [i for i in range(len(x)) if x[i] < x_min or x[i] > x_max]
    gamma[indeces] = 0
    return gamma

def rosso_model(x, x_opt, x_min, x_max):
    gamma = (x - x_max) * (x - x_min)**2 / ((x_opt - x_min) * (x_opt - x_min)* (x - x_opt) - (x_opt - x_max)*(x_opt + x_min - 2*x))
    indeces = [i for i in range(len(x)) if x[i] < x_min or x[i] > x_max]
    gamma[indeces] = 0
    return gamma

def ratkowsky_model(x, x_min, x_max, b, c):
    # Ratkowsky model only includes min temperature and defines the square root of the growth rate
    return (b*(x - x_min)*(1 - exp(c*(x - x_max))))**2

def mcmeekin_model(a_w, a_w_min):
    return (a_w - a_w_min)/(1- a_w_min)

def arrhenius(x,T, E_a):
    gas_const = 8.314446
    T = T + 273.15
    return x*exp(-E_a / (gas_const * T))

# Thermal inactivation models
def davey_model(T,pH,coeffs):
    T = T + 273.15
    return exp(coeffs[0] + coeffs[1]/T + coeffs[2]*pH + coeffs[3]*pH**2)

def mafart_model(T,pH,D_ref,T_ref, zs):
    # Model for D-value - gives inactivation rate in min^{-1}
    logD = log10(D_ref) - (T - T_ref)/zs[0] - (pH-7)**2/(zs[1]**2)
    return log(10)/(10**logD)

def mafart_model_aw(T,pH,aw,D_ref,T_ref, zs):
    logD = log10(D_ref) - (T - T_ref)/zs[0] - (pH-7)**2/(zs[1]**2) - (aw-1)/zs[2]
    return log(10)/(10**logD)

def cerf_model(T,pH,aw,coeffs):
    # model from Cerf,Davey & Sadoudi, 1996 - gives inactivation rate in sec^{-1}
    T = T + 273.15
    return exp(coeffs[0] + coeffs[1]/T + coeffs[2]*pH + coeffs[3]*pH**2 + coeffs[4]*aw**2)

def zweitering_model(T,T_min,T_max,b,c):
    # model for maximum growth rate as function of temperature
    # temperature is in Celcius!
    return b * (1 - exp(c * (T-T_max))) * (T - T_min)**2


def water_activity(moisture_content, T, time):
    # moisture_content - moisture content in parts
    # T - temperature in Kelvin
    # time - time in days
    rate = .01727 + .02613*(.9359**time)
    pow  = .69910 + .41730*(.9434**time)
    T = T + 273.15
    return 1 - exp(-T*rate*(moisture_content**pow))
# ######################################################################################################################
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
    fig1.set_size_inches(13.0, 8.0)
    labels = ['$B.c$', '$C.b$', '$E.c.$', '$F.e$']
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
            axs1[axCol, axRow].set_title('%f days'%(times[j]))
            axs1[axCol, axRow].xaxis.set_ticks(arange(.1, 1.1, .1))
    ax_top = axs1[0, 0].twinx()
    bottom, top = axs1[0, 0].get_ylim()
    ax_top.set_ylim(bottom, top)
    ax_top.set_yticks([.92, .935, .95, .97])
    ax_top.set_yticklabels(labels)
    ax_top = axs1[1, 0].twinx()
    bottom, top = axs1[1, 0].get_ylim()
    ax_top.set_ylim(bottom, top)
    ax_top.set_yticks([.92, .935, .95, .97])
    ax_top.set_yticklabels(labels)
    plt.tight_layout()
    plt.savefig('Figures/aw_vs_moisture.png')
    tkz.save('Tikzes/aw_vs_moisture.tikz')

    # pH inactivation curve - arbitrary values
    phs = arange(4., 10.5, .1)
    temperatures = arange(10., 60., .1)
    rates = cardinal_model(phs, 7.5, 5.1, 9.5)
    rates_ecol = cardinal_model(phs, 7, 4, 10)
    rates_fe   = cardinal_model(phs, 7, 5, 9.6)
    rates_bacil_c = cardinal_model(phs, 7, 4.9, 9.3)
    rates_clostr_b = cardinal_model(phs, 7, 4.6, 9)
    rates_clostr_p = cardinal_model(phs,  7, 5.5, 9.0)
    t_rates_ecol = cardinal_model(temperatures, 40.3, 5.6, 47.3)
    t_rates_fe   = cardinal_model(temperatures, 42, .1, 53.5)
    t_rates_bacil_c = cardinal_model(temperatures, 40.1, 4.1, 50.0)
    t_rates_clostr_b = cardinal_model(temperatures, 39.3, 11.0, 45.8)
    t_rates_clostr_p = cardinal_model(temperatures, 45.0, 10.0, 48.8)
    fig2, axs2 = plt.subplots(1, 2)
    fig2.set_size_inches(13.0, 6.0)
    axs2[0].plot(phs, rates_ecol,  label='$E.coli$')
    axs2[0].plot(phs, rates_bacil_c,  label='$B.cereus$')
    axs2[0].plot(phs, rates_clostr_b,  label='$C.botulinum$')
    axs2[0].plot(phs, rates_clostr_p,  label='$C.perfringens$')
    axs2[0].plot(phs, rates_fe,  label='$E.faecium$')
    axs2[0].legend()
    axs2[0].set_xlabel('pH')
    axs2[0].set_ylabel('$\gamma(\mathrm{pH})$')
    axs2[0].set_title ('Growth rate factor vs pH (cardinal)')
    axs2[1].plot(temperatures, t_rates_ecol,  label='$E.coli$')
    axs2[1].plot(temperatures, t_rates_bacil_c,  label='$B.cereus$')
    axs2[1].plot(temperatures, t_rates_clostr_b,  label='$C.botulinum$')
    axs2[1].plot(temperatures, t_rates_clostr_p, label='$C.perfringens$')
    axs2[1].plot(temperatures, t_rates_fe,  label='$E.faecium$')
    axs2[1].legend()
    axs2[1].set_xlabel('T, $^{\circ}$C')
    axs2[1].set_ylabel('$\gamma(\mathrm{T})$')
    axs2[1].set_title ('Growth rate factor vs T (cardinal)')
    plt.tight_layout()
    plt.savefig('Figures/inhibiting_factors_growth.png')
    tkz.save('Tikzes/inhibiting_factors_growth.tikz')


    # Arrhenius curves for different activation energy values:
    Death_u, Death_v = .1, .2
    tempr_plot = arange(10., 130., 2.)
    energy_plot = arange(1., 5., 1.)
    fig3, axs = plt.subplots(1, 2)
    for e_power in energy_plot:
        energy = 10. ** e_power
        rates_u = arrhenius(Death_u, tempr_plot, energy)
        rates_v = arrhenius(Death_v, tempr_plot, energy)
        axs[0].plot(tempr_plot, log(rates_u), label='$E_a=$ %.3f' % (energy))
        axs[1].plot(tempr_plot, log(rates_v), label='$E_a=$ %.3f' % (energy))
    axs[0].legend()
    axs[0].set_xlabel('T, $^{\circ}$C')
    axs[0].set_ylabel('$\log(r_d)$, for u')
    axs[1].legend()
    axs[1].set_xlabel('T, $^{\circ}$C')
    axs[1].set_ylabel('$\log(r_d)$, for v')
    plt.show()
    plt.savefig('Figures/Arrhenius_death_rates.png')
    tkz.save('Tikzes/Arrhenius_death_rates.tikz')

#     inactivation curves form papers for specific bacteria
    fig4 = plt.figure()
    ax = fig4.add_subplot(111)
    fig4.set_size_inches(10.0, 6.0)
    Death_ecol = cerf_model(tempr_plot, 7, 0.96, [86.49, -.3028 * (10 ** 5), -.5470, .0494, 3.067])
    Death_bacillis = mafart_model_aw(tempr_plot,7,0.96,0.676,100,[9.28, 4.08, 0.164])
    Death_clostr = mafart_model_aw(tempr_plot,7,0.96,0.000000045,100,[7.97, 6.19, 0.125])
    Death_clostr_p = mafart_model_aw(tempr_plot,7,0.96, 0.95, 100, [10.05, 6.19, 0.125]) # clostr perfringens
    Death_fe = mafart_model_aw(tempr_plot,7,0.96,0.796,100,[12.86, 4.19, 0.185])
    temprs_kelvin = 1/(tempr_plot + 273.15)
    ax.plot(temprs_kelvin, Death_ecol*3600, label='$E.coli$')
    ax.plot(temprs_kelvin, Death_bacillis*60, label='$B.cereus$')
    ax.plot(temprs_kelvin, Death_clostr*60, label='$C.botulinum$')
    ax.plot(temprs_kelvin, Death_clostr_p*60, label='$C.perfringens$')
    ax.plot(temprs_kelvin, Death_fe*60, label='$E.faecium$')
    ax_top = ax.twiny()
    ax.set_xlim([temprs_kelvin[-1], temprs_kelvin[0]])
    ax_top.set_xlim([tempr_plot[-1], tempr_plot[0]])  # same limits
    ax_top.set_xlabel('T,$^{\circ}$C')
    # ax_top.set_xticks(flip(tempr_plot[d]))
    # ax_top.set_xticklabels(flip(tempr_plot[d]))
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('1/T, 1/Kelvin')
    ax.set_ylabel('inactivation rate, $h^{-1}$')
    plt.tight_layout()
    plt.savefig('Figures/Cerf_death_rates.png')
    tkz.save('Tikzes/Cerf_death_rates.tikz')

    #   Schoolfield growthrates
    enthalps = array([9963, -51510, 21400])
    temprs = arange(10., 43., 1.)
    a_ws = arange(0.96, 1.01, .01)
    sch_rates_ecol = schoolfield(temprs,enthalps, 316.4, 291.2, 0.273)
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
    fig5 = plt.figure()
    ax = fig5.add_subplot( 111 )
    ax.plot(temprs_kelvin,(growths_ec[4, :]),label='ComBase')
    ax.plot(temprs_kelvin,sch_rates_ecol,label='Schoolfield')
    ax_top = ax.twiny()
    ax_top.set_xlim([temprs[-1], temprs[0]]) # same limits
    ax_top.set_xlabel('T,$^{\circ}$C')
    d = array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.set_xlim([temprs_kelvin[-1], temprs_kelvin[0]])  # same limits
    ax.set_xticks(flip(temprs_kelvin[d]))
    # ax.set_xticklabels(flip(temprs_kelvin[d]))
    ax_top.set_xticks(flip(temprs[d]))
    # ax_top.set_xticklabels(flip(temprs[d]))
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('1/T, 1/Kelvin')
    ax.set_ylabel('growth rate, $h^{-1}$')
    plt.savefig('Figures/Arrhenius_ecoli')
    tkz.save('Tikzes/Arrhenius_ecoli.tikz')

    tempr_core = array([21., 30., 58., 61., 63., 65., 64., 63., 61., 60., \
                        61., 62., 61., 60., 59., 57., 51., 42., 35., 32., \
                        30., 29., 27., 25., 25., 25.])
    days_tempr = array([1, 3, 5, 7, 9, 10, 12, 13, 14, 15, \
                        16, 18, 20, 22, 27, 29, 33, 39, 46, 49, \
                        54, 61, 63, 70, 84, 110])
    ph_compsoting = array([7.8, 8.1, 8.05, 8.0, 7.9, 7.5, \
                           7.5, 7.5, 7.4, 7.4, 7.3, 7.2, 7.2, 7.1, 7.1])
    days_ph = array([1, 3, 6, 8, 14, 21, 28, 35, 42, 49, 56, 63, 70, 84, 110])
    moist_core = array([60., 57., 45., 44., 60., 54., 51., 41., 37., 36., 28., 26., ])
    days_moist = array([1, 8, 14, 21, 35, 42, 49, 56, 63, 70, 84, 110])
    fig6, axs3 = plt.subplots(3, 1)
    fig6.set_size_inches(10.0, 6.0)
    axs3[0].plot(days_tempr,tempr_core)
    axs3[0].set_xlabel('time, days')
    axs3[0].set_ylabel('T, $^{\circ}C$')
    axs3[0].set_title('Core temperature')
    axs3[1].plot(days_ph,ph_compsoting)
    axs3[1].set_xlabel('time, days')
    axs3[1].set_ylabel('pH')
    axs3[1].set_title('Core pH level')
    axs3[2].plot(days_moist,moist_core)
    axs3[2].set_xlabel('time, days')
    axs3[2].set_ylabel('Moisture, $\%$')
    axs3[2].set_title('Moisture content')
    plt.tight_layout()
    plt.savefig('Figures/Composting_environement')
    tkz.save('Tikzes/Composting_environment.tikz')

    fig7 = plt.figure()
    ax = fig7.add_subplot(111)
    fig7.set_size_inches(10.0, 6.0)
    ax.plot(temperatures,zweitering_model(temperatures,.1,53.3,0.0256,0.1436),label="$E.faceum")
    ax.set_xlabel('T, $^{\circ}C$')
    ax.set_xlabel('$\mu_{max}$')
    plt.tight_layout()
    plt.savefig('Figures/fe_max_growth_rate')
    tkz.save('Tikzes/fe_max_growth_rate.tikz')


