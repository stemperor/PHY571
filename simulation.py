from utility import *
import matplotlib.pyplot as plt

# Physical constants
c=299792458 
pi=np.pi
hb= 6.62607015e-34/(2*pi)
e = 1.60217662e-19
me = 9.10938356e-31
eps0 = 8.8541878128e-12


class Environment :
    """Class which stores physical properties and constants of the problem"""
    
    def __init__ (self,K=7,beta=6.5e-104,DT=110e-15, n2=5.57e-23,w=2*pi/775e-9*c,tau=3.5e-13):
        """Inputed physical properties are taken identical to studies articles:

        K: Multiphoton absorption number (K-photon MPI)
        beta: K-multiphoton absorption coefficient
        DT: Laser pulse time length
        n2: Material non linear effect parameter
        w: Light angular frequency
        tau: Electron collision time
        
        Other parameters are calculated from the given data
        """
        
        self.K = K
        self.beta = beta
        self.DT = DT
        self.n2 = n2
        self.w = w
        self.tau=tau

        # Light k-vector
        self.k = w/c
        # Cross section for inverse bremsstrahlung
        self.sigma=(self.k*e**2 * tau/(w*me*eps0))/(1+(w*tau)**2)

        # Equation coefficients (see written report)
        self.rho= beta/(hb*w*2*K**(3/2))*DT*pi**0.5
        self.b= 1j/(2*self.k)
        self.c= -1j*w*tau*self.sigma*self.rho/2 
        self.d= -beta/2 
        self.e= 1j*w*n2/c

        self.Pcr=0.159*(2*pi/w*c)**2/n2
    
    def eq (self,E) :
        """Returns the multiple coefficients of the propagation equation in the form of a dictionary for Runge-Kutta method"""

        E_m = np.absolute(E)

        # Previously, because of overflow errors, calculations were performed logarithmically
        """
        pE = np.log(E_m)
        t1=self.b
        t2=np.exp(np.log(-self.c.imag)+pE*(2*self.K))*(-1j)
        t3=np.exp(np.log(-self.d)+pE*(2*self.K-2))*(-1)
        t4=self.e*E_m**2
        """
        
        t1=self.b
        t2=self.c*E_m**(2*self.K)
        t3=self.d*E_m**(2*self.K-2)
        t4=self.e*E_m**2
        
        return {"diffraction" : t1, "plasma_defocusing" : t2, "MPI" : t3, "nonlinear_focusing" : t4}

    def eq_analytic (self) :
        """Returns the multiple coefficients of the propagation equation in the form of a dictionary for analytical method"""

        t1=self.b
        t2=self.c.imag
        t3=self.d
        t4=self.e.imag
        
        return {"diffraction" : t1, "plasma_defocusing" : t2, "MPI" : t3, "nonlinear_focusing" : t4}

class Field1D:
    """Simulate the evolution of a time-independant 1D field in a nonlinear medium.
    
    __init__: Environment, float, int -> None
        Initiate object with environmental parameters.

    set_gaussian_field: float, float, float -> None
        Set initial field as a gaussian field with possible focalisation.

    step: float, dict -> None
        Calculate the the next field, with an adequately calculated step size. Uses Runge-Kutta

    step_analytical: float, dict -> None
        Calculate the the next field, with an adequately calculated step size. Uses analytical solutions

    total_field_show: float -> None
        Plot field at all calculated steps

    field_show: float -> None
        Plot field at a given step
    
    clear: float -> None
        Remove field data for the smallest time steps (save RAM)

    adaptative_step: float, float -> float
        Calculate optimal step
        """
    

    def __init__(self, env, sim_size, steps_pow):
        """
        env: Environment
        sim_size: float
            Simulation zone will be a square of side 2*sim_size
        steps_pow: float
            the resolution will be 2**steps_pow along each axis"""
        
        self.env = env

        self.dx = sim_size*2 / 2**steps_pow
        self.x = np.linspace(-sim_size,sim_size,2**steps_pow)
        self.nx = len(self.x)

        # Are stored:
        self.fields = [None]    # The fields
        self.maxes = []         # The fields' max values
        self.zs = [0]           # The positions of the calculated fields
        self.dz = []            # The step sizes
        self.ps=[]              # Maximum phase changes at each step
        self.es=[]              # Maximum field changes at each step
        self.widths = []        # Width of the smallest beam filament
        
        
    def set_gaussian_field(self, w0, peak_power_normalized, f0 = None):
        """ Sets initial field to be a gaussian beam, with an optional parabolic phase.

        w0: float
            Standard deviation of gaussian beam
        peak_intensity: float
            Field norm at the center of the gaussian
        f0: float, optional
            proportional to focal length of parabolic phase, default None
        """
        r2 = self.x**2
        peak_intensity=np.sqrt(2*peak_power_normalized*self.env.Pcr/(pi*(w0)**2))
        self.fields[0] = (np.exp(-r2/w0**2)*peak_intensity).astype(complex) # Gaussian beam
        
        """
        perturb = np.random.normal(1, 0.0, self.fields[0].shape)
        perturb[perturb<=0] =0
        self.fields[0] *= perturb"""
        
        self.maxes.append(np.max(np.absolute(self.fields[0])))
        self.widths = [w0]
        
        if f0:
            # Parabolic phase to focus the laser
            self.fields[0]*= np.exp(-1j*r2/f0**2)

    
    def step(self, maxstep = 1e-3, terms = 
             {"diffraction" : True, "plasma_defocusing" : True, "MPI" : True, "nonlinear_focusing" : True}):
        """ Calculates the field using Runge-Kutta method and stores the field at the next step.

        terms: dict
            Set "diffraction", "plasma_defocusing", "MPI" or "nonlinear_focusing" to True or False to respectively 
            enable and disable said terms in the field calculation
        """
            
        if self.fields[-1] is None:
            raise ValueError("No input field.")

        dz = self.adaptative_step(maxstep=maxstep)

        # The following algorithm is an implementation of a split scheme using:
        #   - Runge-Kutta fourth order for the "plasma_defocusing", "MPI" and "nonlinear_focusing" terms
        #   - The Fourier method for the diffraction's laplacian's term


        equ_terms = self.env.eq(self.fields[-1])
        
        #### Fourier method (see report)

        if terms["diffraction"]:

            field_ft = np.fft.fft(self.fields[-1])
            freqx = np.fft.fftfreq(self.nx, self.dx)
            k_square = freqx**2*4*pi**2

            field_ft *= np.exp(-k_square*dz*equ_terms["diffraction"])
            
            self.fields.append(np.fft.ifft(field_ft))

        else:
            self.fields.append(self.fields[-1])

        #### Runge-Kutta method (see report)
        
        k1 = (equ_terms["plasma_defocusing"]*terms["plasma_defocusing"] +
                       equ_terms["MPI"]*terms["MPI"]+ 
                       equ_terms["nonlinear_focusing"]*terms["nonlinear_focusing"])*self.fields[-1]
        
        
        f2 = self.fields[-1] + k1/2*dz
        equ_terms2 = self.env.eq(f2)
        
        k2 = (equ_terms2["plasma_defocusing"]*terms["plasma_defocusing"] +
                       equ_terms2["MPI"]*terms["MPI"]+ 
                       equ_terms2["nonlinear_focusing"]*terms["nonlinear_focusing"])*f2
        

        f3 = self.fields[-1] + k2/2*dz
        equ_terms3 = self.env.eq(f3)
        
        k3 = (equ_terms3["plasma_defocusing"]*terms["plasma_defocusing"] +
                       equ_terms3["MPI"]*terms["MPI"]+ 
                       equ_terms3["nonlinear_focusing"]*terms["nonlinear_focusing"])*f3
        

        f4 = self.fields[-1] + k3*dz
        equ_terms4 = self.env.eq(f4)
        
        k4 = (equ_terms4["plasma_defocusing"]*terms["plasma_defocusing"] +
                       equ_terms4["MPI"]*terms["MPI"]+ 
                       equ_terms4["nonlinear_focusing"]*terms["nonlinear_focusing"])*f4
        

        self.fields[-1] += 1.0/6 * dz * (k1+2*(k2+k3)+k4)


        #### Stores other information

        self.zs.append(self.zs[-1] + dz)
        self.dz.append(dz)
        
        field_abs = np.absolute(self.fields[-1])
        maxpos = np.argmax(field_abs)
        m = field_abs[maxpos]
        self.maxes.append(m)
        
        ##### FIND W0
        
        in_peak = field_abs>m/2 # Find the half height zones of the field

        # Not so interesting technical details to find the half width maximum
        zones = true_zones(in_peak)
        fwhm = None
        for i in zones:
            if i[0] <= maxpos <= i[1]:
                fwhm = i[1] - i[0]
                
        if fwhm is None:
            print("Failed to find filament")
            self.widths.append(-1)
        else:
            # the 2.355 factor links fwhm to gaussian standard deviation
            self.widths.append(fwhm/2.355 * self.dx*np.sqrt(2))


    def step_analytic(self, maxstep = 1e-3, terms = 
             {"diffraction" : True, "plasma_defocusing" : True, "MPI" : True, "nonlinear_focusing" : True}):
        """ Calculates and stores the field at the next step analytically.

        terms: dict
            Set "diffraction", "plasma_defocusing", "MPI" or "nonlinear_focusing" to True or False to respectively 
            enable and disable said terms in the field calculation
        """
            
        if self.fields[-1] is None:
            raise ValueError("No input field.")

        dz = self.adaptative_step(maxstep=maxstep)

        # The following algorithm is an implementation of a split scheme using:
        #   - An analytical (technically third order) solution for the "plasma_defocusing", "MPI" and "nonlinear_focusing" terms
        #   - The Fourier method for the diffraction's laplacian's term


        equ_terms = self.env.eq_analytic()
        
        #### Fourier method (see report)

        if terms["diffraction"]:

            field_ft = np.fft.fft(self.fields[-1])
            freqx = np.fft.fftfreq(self.nx, self.dx)
            k_square = freqx**2*4*pi**2

            field_ft *= np.exp(-k_square*dz*equ_terms["diffraction"])
            
            self.fields.append(np.fft.ifft(field_ft))

        else:
            self.fields.append(self.fields[-1])

        #### Analytical approach (see report)

        K = float(self.env.K)

        field_abs = np.absolute(self.fields[-1])
        field_abs2 = field_abs**2

        X = dz*equ_terms["MPI"]*(2-2*K)/field_abs2**(1-K) # See report

        # Norm of field at the next step
        field_norm = (1+X)**(1/(2-2*K))*field_abs

        # Just to improve readability
        fact = -1/(2*equ_terms["MPI"]*(K-2)*(1-K))
        fact1 = fact*(K**2 - 3*K + 2)
        fact2 = fact*(1-K)

        term1=0
        term2=0

        


        # Calculation of E^p - E0^p
        # This is technically an approximation, but the errors made are smaller than the floating point precision
        # of our machines. We can thus consider these calculations exact.

        # See report for explanation for this threshold
        thresh = 1e-5

        X2 = X**2
        X3 = X**3

        # Values of exponents
        p1 = -1/(K-1)
        p2 = (K-2)/((K-1))

        field_norm2 = field_norm**2

        # Two ways of calculating: direct calculations (dE1_calc) and series expansion (dE1_approx)
        dE1_calc = field_norm2 - field_abs2
        dE1_approx = field_abs2 * (p1*X + p1*(p1-1)*X2/2 + p1*(p1-1)*(p1-2)/6*X3)
        
        field_norm22_K = field_norm2**(2-K)

        # Same thing for p = p2
        dE2_calc = field_norm22_K - field_abs2**(2-K)
        dE2_approx = field_abs2**(2-K) * (p2*X + p2*(p2-1)*X2/2 + p2*(p2-1)*(p2-2)/6*X3)

        # Depending on the value of X, choose a method of calculation.
        dE1 = np.where(X<thresh, dE1_approx, dE1_calc)
        dE2 = np.where(X<thresh, dE2_approx, dE2_calc)

    
        if terms["plasma_defocusing"]:
            
            term1 = equ_terms["plasma_defocusing"]*fact1*dE1

        
        if terms["nonlinear_focusing"]:
            term2 = equ_terms["nonlinear_focusing"]*fact2*dE2
    
        phi = np.angle(self.fields[-1]) + term1+term2
        self.fields[-1] = field_norm*np.exp(1j*phi)  
       

        #### Stores other information

        self.zs.append(self.zs[-1] + dz)
        self.dz.append(dz)
        
        field_abs = np.absolute(self.fields[-1])
        maxpos = np.argmax(field_abs)
        m = field_abs[maxpos]
        self.maxes.append(m)
        
        ##### FIND W0
        
        in_peak = field_abs>m/2 # Find the half height zones of the field

        # Not so interesting technical details to find the half width maximum
        zones = true_zones(in_peak)
        fwhm = None
        for i in zones:
            if i[0] <= maxpos <= i[1]:
                fwhm = i[1] - i[0]
                
        if fwhm is None:
            print("Failed to find filament")
            self.widths.append(-1)
        else:
            # the 2.355 factor links fwhm to gaussian standard deviation
            self.widths.append(fwhm/2.355 * self.dx*np.sqrt(2))


        
    def field_show(self, dist):
        """ Given a distance to the origin, plot the estimated field

        The field returned will be the furthest field calculated at a shorter distance than dist
        """

        index = np.searchsorted(self.zs, dist, side="right") - 1

        plt.figure()
        print("Distance to origin:",self.zs[index],"m")
        plt.plot(self.x,np.absolute(self.fields[index])**2)
        plt.xlabel(r"$x$ $(mm)$")
        plt.ylabel(r"Field intensity (W.m^{-2})")

    def total_field_show(self, step, x_select = None, x_compress = 1):
        """Show the complete propagation of the field as an image.
        
        step: float
            For all fields closer than step to one another, a single one will be kept for representation
        x_select: float
            The field will only be shown between [-x_select, +x_select], Defaults to the whole simulation size.
        x_compress: float
            The shown resolution of the field will be self.nx/x_compress. Defaults to 1.
        """

        fields = []

        # Choose the fields to show
        i = 0
        n_steps = 0
        last_z = 0
        while i < len(self.fields):
            if self.zs[i] >= step*n_steps:
                fields.append(self.fields[i])
                n_steps += 1
                last_z = self.zs[i]
            i+=1

        # Make the desired image
        size = self.x[-1]
        if x_select is not None:
            cut_dist = max(self.x[-1] - x_select,  0)/self.dx
            im = np.array(fields)[::, cut_dist:-cut_dist:x_compress]
            size = min(self.x[-1], cut_dist)
        else:
            im = np.array(fields)[::, ::x_compress]

        plt.figure()
        plt.imshow(np.absolute(im.T)**2, extent = [0, last_z, -size, size], aspect="auto")
        plt.colorbar(label=r"Field intensity ($W.m^{-2}$)")
            
        plt.xlabel(r"Distance ($m$)")
        plt.ylabel(r"$x$ position ($m$)")
        
    def clear(self, step):
        """Clears redundant fields for a given step size.
        
        That is, if multiples calculated fields are at a distance less than step, only one is kept."""

        i = 0
        n_steps = 0
        
        while i < len(self.fields):
            if self.zs[i] < step*n_steps:
                del self.zs[i]
                del self.fields[i]
                del self.maxes[i]
                del self.widths[i]
                del self.ps[i]
                del self.es[i]
                if i>0:
                    del self.dz[i-1]
            else:
                i += 1
                n_steps += 1
                
    def adaptative_step(self,minstep=0,maxstep=1e-3):
        """Calculates a adapted step size based on a given min and max step and the current field"""

        mxfield=self.maxes[-1]
        eqterms=self.env.eq(mxfield)
        
        # See report for details

        energy_step = 2*eqterms["MPI"]
        energy_step = np.absolute(1/energy_step/10)
        self.es.append(energy_step)
        
        phase_step = eqterms["plasma_defocusing"]+eqterms["nonlinear_focusing"]
        phase_step = np.absolute(2*pi/10/phase_step)
        self.ps.append(phase_step)
        
        return max(minstep,min(phase_step,energy_step,maxstep))


class Field:
    """Simulate the evolution of a time-independant 2D field in a nonlinear medium.
    
    __init__: Environment, float, int -> None
        Initiate object with environmental parameters.

    set_gaussian_field: float, float, float -> None
        Set initial field as a gaussian field with possible focalisation.

    step: float, dict -> None
        Calculate the the next field, with an adequately calculated step size. Uses Runge-Kutta

    step_analytical: float, dict -> None
        Calculate the the next field, with an adequately calculated step size. Uses analytical solutions.

    field_show: int -> None
        Show field at a given step
    
    clear: float -> None
        Remove field data for the smallest time steps (save RAM)

    adaptative_step: float, float -> float
        Calculate optimal step
        """
    
    def __init__(self, env, sim_size, steps_pow):
        """
        env: Environment
        sim_size: float
            Simulation zone will be a square of side 2*sim_size
        steps_pow: float
            the resolution will be 2**steps_pow along each axis"""

        self.env = env
        self.steps_pow = steps_pow

        self.xmin, self.xmax = -sim_size,sim_size
        self.ymin, self.ymax = -sim_size,sim_size
        self.dx = sim_size*2 / 2**steps_pow
        self.dy = self.dx
        
        xs = np.linspace(-sim_size,sim_size, 2**steps_pow)
        ys = np.linspace(-sim_size,sim_size, 2**steps_pow)
        
        self.x, self.y = np.ix_(xs, ys)

        self.nx = len(xs)
        self.ny = len(ys)
        
        # Are stored:
        self.fields = [None]
        self.maxes = []
        self.zs = [0]
        self.dz = []
        self.ps = []
        self.widths = []

        # Useful calculations
                
        
        freqx_ = np.fft.fftfreq(self.nx, self.dx)
        freqy_ = np.fft.fftfreq(self.ny, self.dy)
        freqx, freqy = np.ix_(freqx_, freqy_)
        self.k_square = (freqx**2 + freqy**2)*4*pi**2


    def set_gaussian_field(self, w0, peak_power_normalized, f0 = None):
        """ Sets initial field to be a gaussian beam, with an optional parabolic phase.

        w0: float
            Standard deviation of gaussian beam
        peak_intensity: float
            Field norm at the center of the gaussian
        f0: float, optional
            TO BE CHANGED: "focal length" of parabolic phase, default None
        """
        r2 = self.x**2 + self.y**2
        peak_intensity=np.sqrt(2*peak_power_normalized*self.env.Pcr/(pi*(w0)**2))
        self.fields[0] = (np.exp(-r2/w0**2)*peak_intensity).astype(complex)
        
        self.maxes.append(np.max(np.absolute(self.fields[0])))
        self.widths = [w0]
        
        if f0:
            # Parabolic phase to focus the laser
            self.fields[0]*= np.exp(-1j*r2/f0**2)


    def step_analytic(self, maxstep = 1e-3, terms = 
             {"diffraction" : True, "plasma_defocusing" : True, "MPI" : True, "nonlinear_focusing" : True}):
        """ Calculates and stores the field at the next step.

        terms: dict
            Set "diffraction", "plasma_defocusing", "MPI" or "nonlinear_focusing" to True or False to respectively 
            enable and disable said terms in the field calculation
        """

        if self.fields[-1] is None:
            raise ValueError("No input field.")
        
        dz=self.adaptative_step(maxstep=maxstep)
        
        # The following algorithm is an implementation of a split scheme using:
        #   - Runge-Kutta fourth order for the "plasma_defocusing", "MPI" and "nonlinear_focusing" terms
        #   - The Fourier method for the diffraction's laplacian's term
            
        equ_terms = self.env.eq_analytic()

        #### Fourier method (see report)
        
        if terms["diffraction"]:

            field_ft = np.fft.fft2(self.fields[-1])
            field_ft *= np.exp(-self.k_square*equ_terms["diffraction"]*dz)

            self.fields.append(np.fft.ifft2(field_ft))
        else:
            self.fields.append(self.fields[-1])

        #### Analytical approach (see report)

        # See identical commented version in Field1D.step_analytical()

        K = float(self.env.K)

        field_abs = np.absolute(self.fields[-1])
        field_abs2 = field_abs**2

        X = dz*equ_terms["MPI"]*(2-2*K)/field_abs2**(1-K) # See report

        temp_calc = (1+X)**(1/(2-2*K))
        field_norm = temp_calc*field_abs

        fact = -1/(2*equ_terms["MPI"]*(K-2)*(1-K))
        fact1 = fact*(K**2 - 3*K + 2)
        fact2 = fact*(1-K)

        term1=0
        term2=0

        X2 = X**2
        X3 = X**3

        p1 = -1/(K-1)
        p2 = (K-2)/((K-1))

        thresh = 1e-2

        field_norm2 = field_norm**2

        dE1_calc = field_norm2 - field_abs2
        dE1_approx = field_abs2 * (p1*X + p1*(p1-1)*X2/2 + p1*(p1-1)*(p1-2)/6*X3)
        
        field_norm22_K = field_norm2**(2-K)

        dE2_calc = field_norm22_K - field_abs2**(2-K)
        dE2_approx = field_abs2**(2-K) * (p2*X + p2*(p2-1)*X2/2 + p2*(p2-1)*(p2-2)/6*X3)


        dE1 = np.where(X<thresh, dE1_approx, dE1_calc)
        dE2 = np.where(X<thresh, dE2_approx, dE2_calc)

        # Précision du float insuffisante !
    
        if terms["plasma_defocusing"]:
            
            term1 = equ_terms["plasma_defocusing"]*fact1*dE1

        
        if terms["nonlinear_focusing"]:
            term2 = equ_terms["nonlinear_focusing"]*fact2*dE2
    
        phi = np.angle(self.fields[-1]) + term1+term2
        self.fields[-1] = field_norm*np.exp(1j*phi)
        #### Store other information

        self.zs.append(self.zs[-1] + dz)
        self.dz.append(dz)
        
        field_abs = np.absolute(self.fields[-1])
        maxpos = np.argmax(field_abs)
        max_ij = np.unravel_index(maxpos, field_abs.shape)
        m = field_abs[max_ij[0], max_ij[1]]
        self.maxes.append(m)

        ##### FIND W0 

        # Technical details really. It isn't perfect but works for gaussian-like beams
        # See Field1D.step() for simpler version

        in_peak_i = field_abs[max_ij[0]]>m/2
        zones = true_zones(in_peak_i)

        fwhm = None
        for i in zones:
            if i[0] <= max_ij[0] <= i[1]:
                fwhm = i[1] - i[0]
                
        if fwhm is None:
            print("Failed to find filament")
            self.widths.append(-1)
        else:
            self.widths.append(fwhm/2.355 * self.dx*np.sqrt(2))
    
    def step(self, maxstep = 1e-3, terms = 
             {"diffraction" : True, "plasma_defocusing" : True, "MPI" : True, "nonlinear_focusing" : True}):
        """ Calculates and stores the field at the next step.

        terms: dict
            Set "diffraction", "plasma_defocusing", "MPI" or "nonlinear_focusing" to True or False to respectively 
            enable and disable said terms in the field calculation
        """

        if self.fields[-1] is None:
            raise ValueError("No input field.")
        
        dz=self.adaptative_step(maxstep=maxstep)
        
        # The following algorithm is an implementation of a split scheme using:
        #   - Runge-Kutta fourth order for the "plasma_defocusing", "MPI" and "nonlinear_focusing" terms
        #   - The Fourier method for the diffraction's laplacian's term
            
        equ_terms = self.env.eq(self.fields[-1])

        #### Fourier method (see report)
        
        if terms["diffraction"]:

            field_ft = np.fft.fft2(self.fields[-1])
            field_ft *= np.exp(-self.k_square*equ_terms["diffraction"]*dz)

            self.fields.append(np.fft.ifft2(field_ft))
        else:
            self.fields.append(self.fields[-1])

        #### Runger-Kutta method (see report)

        
        k1 = (equ_terms["plasma_defocusing"]*terms["plasma_defocusing"] +
                       equ_terms["MPI"]*terms["MPI"]+ 
                       equ_terms["nonlinear_focusing"]*terms["nonlinear_focusing"])*self.fields[-1]
        
        
        f2 = self.fields[-1] + k1/2*dz
        equ_terms2 = self.env.eq(f2)
        
        k2 = (equ_terms2["plasma_defocusing"]*terms["plasma_defocusing"] +
                       equ_terms2["MPI"]*terms["MPI"]+ 
                       equ_terms2["nonlinear_focusing"]*terms["nonlinear_focusing"])*f2
        

        f3 = self.fields[-1] + k2/2*dz
        equ_terms3 = self.env.eq(f3)
        
        k3 = (equ_terms3["plasma_defocusing"]*terms["plasma_defocusing"] +
                       equ_terms3["MPI"]*terms["MPI"]+ 
                       equ_terms3["nonlinear_focusing"]*terms["nonlinear_focusing"])*f3
        

        f4 = self.fields[-1] + k3*dz
        equ_terms4 = self.env.eq(f4)
        
        k4 = (equ_terms4["plasma_defocusing"]*terms["plasma_defocusing"] +
                       equ_terms4["MPI"]*terms["MPI"]+ 
                       equ_terms4["nonlinear_focusing"]*terms["nonlinear_focusing"])*f4
        

        self.fields[-1] += 1.0/6 * dz * (k1+2*(k2+k3)+k4)
        

        #### Store other information

        self.zs.append(self.zs[-1] + dz)
        self.dz.append(dz)
        
        field_abs = np.absolute(self.fields[-1])
        maxpos = np.argmax(field_abs)
        max_ij = np.unravel_index(maxpos, field_abs.shape)
        m = field_abs[max_ij[0], max_ij[1]]
        self.maxes.append(m)

        ##### FIND W0
        # Technical details really. It isn't perfect but works for gaussian-like beams
        # See Field1D.step() for simpler version
        in_peak_i = field_abs[max_ij[0]]>m/2 # Find the half height zones of the field
        zones = true_zones(in_peak_i)

        fwhm = None
        for i in zones:
            if i[0] <= max_ij[0] <= i[1]:
                fwhm = i[1] - i[0]
                
        if fwhm is None:
            print("Failed to find filament")
            self.widths.append(-1)
        else:
            self.widths.append(fwhm/2.355 * self.dx*np.sqrt(2))
        
    def field_show(self, dist):
        """ Given a distance to the origin, show the estimated field

        The field returned will be the furthest field calculated at a shorter distance than dist
        """

        index = (np.searchsorted(self.zs, dist, side="right") - 1)

        plt.figure()
        print("Distance to origin:",self.zs[index],"m")
        plt.imshow(np.absolute(self.fields[index])**2, extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        plt.xlabel(r"$x$ $(m)$")
        plt.xlabel(r"$y$ $(m)$")
        plt.colorbar(label = r"Field intensity ($W.m^{-2}$)")
        

    def clear(self, step):
        """Clears redundant fields for a given step size.
        
        That is, if multiples calculated fields are at a distance less than step, only one is kept."""
        i = 0
        n_steps = 0
        
        while i < len(self.fields)-1:
            if self.zs[i] < step*n_steps:
                del self.zs[i]
                del self.fields[i]
                del self.maxes[i]
                if i>0:
                    del self.dz[i-1]
            else:
                i += 1
                n_steps += 1
    
    def adaptative_step(self,minstep=0,maxstep=1e-3):
        """Calculates a adapted step size based on a given min and max step and the current field"""

        mxfield=self.maxes[-1]
        eqterms=self.env.eq(mxfield)
        
        energy_step = 2*eqterms["MPI"]
        energy_step = np.absolute(1/energy_step/10)
        
        phase_step = eqterms["plasma_defocusing"]+eqterms["nonlinear_focusing"]
        phase_step = np.absolute(2*pi/10/phase_step)
        
        step=min(phase_step,maxstep,energy_step)
        step=max(minstep,step)
        return(step)
        

# Practical save and load functions


def save_field(field, path):
    save_dict = {}

    save_dict["sim_size"] = field.xmax
    save_dict["steps_pow"] = field.steps_pow
    save_dict["fields"] = field.fields
    save_dict["maxes"] = field.maxes
    save_dict["dz"] = field.dz
    save_dict["zs"] = field.zs
    save_dict["ps"] = field.ps
    save_dict["widths"] = field.widths
    np.save(path, save_dict)


def load_field(env, path):
    save_dict = np.load(path, allow_pickle=True).item()
    field = Field(env, save_dict["sim_size"], save_dict["steps_pow"])

    field.fields = save_dict["fields"]
    field.maxes = save_dict["maxes"]
    field.dz = save_dict["dz"]
    field.zs = save_dict["zs"]
    field.ps = save_dict["ps"]
    field.widths = save_dict["widths"]

    return field




