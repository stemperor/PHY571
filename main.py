from simulation import *
import matplotlib
matplotlib.use('Agg')

env = Environment()

field = Field(env, 2.5e-3, 11)
field.set_gaussian_field(0.7e-3, 4)

for i in range(1000):
    field.step_analytic({"diffraction" : True, "plasma_defocusing" : True, "MPI" : True, "nonlinear_focusing" : True})
    
    if i%20 == 0:
        plt.imshow(np.absolute(field.fields[-1])**2, extent=[field.xmin, field.xmax, field.ymin, field.ymax])
        plt.colorbar(label=r"Field intensity (W.m^{-2})")
        plt.xlabel(r"x (m)")
        plt.ylabel(r"y (m)")
        plt.savefig("figs\\{}.png".format(i))

    if i%50 == 0:
        field.clear(1e-2)
        print(i)


save_field(field, "figs\\field.npy")

        
        




#field.total_field_show(5e-3)

#plt.show(block=True)
