import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters, Minimizer,fit_report
from ScatteringMatrix import ComputeRT
from scipy.interpolate import CubicSpline

# path = "G:/My Drive/20211228 AFM/Python Notebooks/20230511/20230511-cdrive/20230512 Light Reflectometry/using/"
plt.rcParams['figure.figsize'] = [7, 5]
# define fitting wavelength range
start = 350
end = 900

#load nk data base
nk_df = pd.read_csv("Refraction_Index_Library.csv", delimiter = ',', header = 0, skipinitialspace = True)

#load exp data
df = pd.read_csv("SiO2-uv.txt",delimiter = '\t')
wl = df.loc[(df['wl']>=start) & (df['wl']<= end),'wl']
refl = df.loc[(df['wl']>=start) & (df['wl']<= end),'R']
refl/=100
wl_exp = np.array(wl)
R_exp = np.array(refl)
T_exp = np.zeros(len(refl))

plt.scatter(wl_exp,R_exp,s = 1,color = 'red')
plt.scatter(wl_exp,T_exp,s = 1,color = 'red')
plt.show()


#define dispersion models
def cauchy(x,a,b,c):
    
    return a + (b/(x)**2) + (c/(x)**4)
# *using
def sellmeier(x,b1,c1,b2,c2,b3,c3):
    
    return (1+((b1*(x**2))/((x**2)-(c1**2))) + ((b2*(x**2))/((x**2)-(c2**2))) + ((b3*(x**2))/((x**2)-(c3**2))))**(1/2)

def sellmeier_s(x,a,b1,c1,b2,c2):
    
    return (a+((b1*((x)**2))/(((x)**2)-c1))+ ((b2*((x)**2))/(((x)**2)-c2)))**(1/2)

# *using
def cauchy_k(x,ak,bk,ck):
    
    return ak + (bk/(x)**2) + (ck/(x)**4)

#define objective function that return array of residuals to be minimized
def totLossRT_tl(params,x,data = None,return_RT = False):
    
    t = params['t']
#     a = params['a']
#     b = params['b']
#     c = params['c']

#     a = params['a']
    b1 = params['b1']
    c1 = params['c1']
    b2 = params['b2']
    c2 = params['c2']
    b3 = params['b3']
    c3 = params['c3']
    
    ak = params['ak']
    bk = params['bk']
    ck = params['ck']
    
    def cal_RT():
        
        Phi = 5
        LayerStructure = [
            ['Air',0,False],
            ['SiO2',989.4,False],
            ['Si',1000000,True],
            ['Air',0,False],
        ]
        structure = []
        for layer_i in range(len(LayerStructure)-1):
            mat_label = LayerStructure[layer_i][0]
            wl = nk_df[mat_label + '_wl'] + 0.0001
            n = nk_df[mat_label + '_n']
            k = nk_df[mat_label + '_k']
#             wl/=1000
            wl = np.array(wl)
            n = np.array(n)
            k = np.array(k)
            n_interp = np.interp(wl_exp,wl,n)
            k_interp = np.interp(wl_exp,wl,k)
            Thickness = LayerStructure[layer_i][1]
            Roughness = 0
            Incoherence = LayerStructure[layer_i][2]
            Ref_idx = n_interp + 1j*k_interp
            if layer_i == 0 or layer_i == len(LayerStructure)-1:
                Thickness = 1000000
                Incoherence = False
            structure.append([Thickness, Ref_idx,Incoherence,Roughness])
            
        #thin film of concern
        layer1 = True
        n_layer1 = sellmeier(wl_exp,b1,c1,b2,c2,b3,c3) #n dispersion layer1
        k_layer1 = cauchy_k(wl_exp,ak,bk,ck) #k dispersion layer1
        if layer1 == True:
            n_interp = n_layer1
            k_interp = k_layer1
            Ref_idx = n_interp + 1j*k_interp
            Thickness = t
            Incoherence = False
            Roughness = 0
            structure[1] = [Thickness, Ref_idx, Incoherence, Roughness]
            R_cal, T_cal = ComputeRT(structure,wl_exp,(Phi/180)*np.pi)
            
        return R_cal,T_cal
    
    R_cal, T_cal = cal_RT()
    Loss = np.dot(R_cal-R_exp,R_cal-R_exp) + np.dot(T_cal-T_exp,T_cal-T_exp)
    
    if not return_RT:
        return R_cal-R_exp
    else:
        try:
            return R_cal, T_cal
        except:
            R_cal,T_cal = cal_RT()
            return R_cal, T_cal

pars = Parameters()
#set up the initial parameter values
pars.add('t',value = 900, min = 0,max= 1100,brute_step = 1)
# pars.add('a',value = 1.15, min = 1.2,max = 4)
# pars.add('b',value = 0.005, min =0 ,max = 0.1)
# pars.add('c',value = 1e-3, min = 0,max = 1)

# pars.add('a',value = 1.65, min = 1,max = 3)
pars.add('b1',value = 1, min = 1,max = 3,brute_step = 0.01)
pars.add('c1',value = 0.065, min = 0, max =20,brute_step = 0.001) 
pars.add('b2',value = 0.009, min = 0.2,max = 5,brute_step = 0.001) 
pars.add('c2',value = 0.05, min = 0,max = 20,brute_step = 0.25) 
pars.add('b3',value = 0.2, min = 0.2,max = 5,brute_step = 0.1)  
pars.add('c3',value = 0.01, min = 0,max = 20,brute_step = 0.01) 

# pars.add('a',value = 1.22, min = 1,max = 3)
# pars.add('b1',value = 0.4, min =0,max = 3)
# pars.add('c1',value = 0.065, min =0 ,max =20)
# pars.add('b2',value = 0.009, min = 0,max = 3)
# pars.add('c2',value = 5e-5, min = 0,max = 20)

pars.add('ak',value = 0.01, min = 0,max = 1)
pars.add('bk',value = 7e-4, min = 0,max = 1)
pars.add('ck',value = 5e-6, min = 0,max = 5e-5)

# pars.pretty_print()

#plot the initial guess
R_cal,T_cal = totLossRT_tl(pars, x = wl_exp,return_RT = True)
plt.plot(wl_exp,R_cal,label = 'model',color = 'black')
plt.scatter(wl_exp,R_exp,s = 3,label = 'data', color = 'red')
plt.legend()
plt.show()       

# Apply Minimizer for optimization
fitter = Minimizer(totLossRT_tl,pars,fcn_args = (wl_exp,R_exp), nan_policy = 'propagate')
result = fitter.minimize()
final = R_exp + result.residual
final = np.array(final)

# plot the fitting
plt.scatter(wl_exp,R_exp,s = 10,color = 'red',label = 'data',alpha =0.7)
plt.plot(wl_exp,final,color = 'black',label = 'bestfit')
plt.legend()
plt.show()

# extract the optimized parameters
t = result.params['t'].value

# for cauchy model
# a = result.params['a'].value
# b = result.params['b'].value
# c = result.params['c'].value

# for sellmeier model
# a = result.params['a'].value
b1 = result.params['b1'].value
c1 = result.params['c1'].value
b2 = result.params['b2'].value
c2 = result.params['c2'].value
b3 = result.params['b3'].value
c3 = result.params['c3'].value

# for cauchy extinction model
ak = result.params['ak'].value
bk = result.params['bk'].value
ck = result.params['ck'].value



# define the dispersion model using the optimized parameters
# n1_fit = cauchy(wl_exp,a,b,c)
n1_fit = sellmeier(wl_exp,b1,c1,b2,c2,b3,c3) #fitted n dispersion
k1_fit = cauchy_k(wl_exp,ak,bk,ck) #fitted k dispersion
n1_fit = np.array(n1_fit)
k1_fit = np.array(k1_fit)
miu = np.exp((-4*np.pi*k1_fit*t)/wl_exp)

# extract si n and k 
wl2 = nk_df.loc[(nk_df['Si_wl']>=start) & (nk_df['Si_wl']<=end),'Si_wl']
n2 = nk_df.loc[(nk_df['Si_wl']>=start) & (nk_df['Si_wl']<=end),'Si_n']
k2 = nk_df.loc[(nk_df['Si_wl']>=start) & (nk_df['Si_wl']<=end),'Si_k']

# interpolate si n and k to match data points
nsi = CubicSpline(wl2,n2)
ksi = CubicSpline(wl2,k2)

n2 = nsi(wl_exp)
k2 = ksi(wl_exp)

# upper and lower envelope
r01 = ((1-n1_fit)-k1_fit)/((1+n1_fit)+k1_fit)
r12 = ((n2-n1_fit)+(k2-k1_fit))/((n2+n1_fit)+(k2+k1_fit))

upper_env = ((r01+miu*r12)/(1+miu*r01*r12))**2
lower_env = ((r01-miu*r12)/(1-miu*r01*r12))**2

plt.scatter(wl_exp,R_exp,s = 10,color = 'red',label = 'data',alpha =0.7)
plt.plot(wl_exp,final,color = 'black',label = 'bestfit')
plt.plot(wl_exp,upper_env,'--',color = 'cyan',label = 'upper')
plt.plot(wl_exp,lower_env,'--',color = 'cyan',label = 'lower')
plt.legend()
plt.show()

lam = np.linspace(start, end, num = len(wl_exp))
plt.plot(lam,n1_fit)
plt.show()

# print fitted results (fit report)
print(f"thickness:{result.params['t'].value:.2f}")
print(fit_report(result))