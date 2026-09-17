"""
Rotating Detonation Engine (RDE) Combustor Element for pyCycle.

Implements the pressure-gain combustion physics of an RDE, including:
1. Injector plenum feed pressure drop
2. Humphrey / Fickett-Jacobs detonation thermodynamics
3. Detonation wave kinematics (CJ wave speed and rotation frequency)
4. Unsteady-to-steady flow realization efficiency
5. Integration with pyCycle Element API (ThermoAdd + Thermo)
"""

import numpy as np
import openmdao.api as om

from pycycle.constants import R_UNIVERSAL_ENG, g_c
from pycycle.thermo.thermo import Thermo, ThermoAdd
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element


class RDEPressureGainComp(om.ExplicitComponent):
    """
    Computes the stagnation pressure gain and wave kinematics across an RDE annulus.

    Governing Equations:
    --------------------
    1. Injector Plenum Pressure:
       Pt_inj = Pt_in * (1.0 - dPqP_inj)

    2. Fuel Energy Release:
       q = (FAR * Q_fuel) / (1.0 + FAR)  [Btu/lbm]

    3. Gas Properties:
       R_gas = R_UNIVERSAL_ENG / MW_gas  [Btu/(lbm*degR)]
       cv = R_gas / (gamma_gas - 1.0)

    4. Humphrey Pressure Ratio (Ideal Constant-Volume Combustion):
       Pi_ideal = 1.0 + q / (cv * Tt_in) = 1.0 + (gamma_gas - 1.0) * q / (R_gas * Tt_in)

    5. Realized Stagnation Pressure Gain:
       Pi_det = 1.0 + eta_rde * (Pi_ideal - 1.0)
       Pt_out = Pt_inj * Pi_det = Pt_in * (1.0 - dPqP_inj) * Pi_det
       PR_RDE = Pt_out / Pt_in = (1.0 - dPqP_inj) * Pi_det

    6. Wave Kinematics:
       M_cj = sqrt(1.0 + (gamma_gas**2 - 1.0)/(2.0*gamma_gas) * q/(R_gas*Tt_in)) + ...
       D_cj = M_cj * sqrt(gamma_gas * R_gas * 778.169 * g_c * Tt_in)  [ft/s]
       f_rde = (N_waves * D_cj) / (pi * (dia_annulus / 12.0))         [Hz]
    """

    def initialize(self):
        self.options.declare('detonation_mode', default='HUMPHREY', values=['HUMPHREY', 'CJ'],
                             desc='Thermodynamic mode for detonation pressure rise')

    def setup(self):
        # Stagnation flow inputs from upstream
        self.add_input('Pt_in', val=100.0, units='lbf/inch**2', desc='Inlet total pressure')
        self.add_input('Tt_in', val=1000.0, units='degR', desc='Inlet total temperature')
        self.add_input('FAR', val=0.03, desc='Fuel to air ratio')

        # Combustor design parameters
        self.add_input('dPqP_inj', val=0.12, desc='Injector feed pressure loss fraction')
        self.add_input('eta_rde', val=0.85, desc='Detonation combustor realization efficiency')
        self.add_input('Q_fuel', val=18600.0, units='Btu/lbm', desc='Lower heating value of fuel')
        self.add_input('gamma_gas', val=1.33, desc='Effective ratio of specific heats')
        self.add_input('MW_gas', val=28.96, units='lbm/mol', desc='Molecular weight of gas mixture')
        self.add_input('dia_annulus', val=12.0, units='inch', desc='Mean diameter of RDE annulus')
        self.add_input('N_waves', val=1.0, desc='Number of co-rotating detonation waves')

        # Performance outputs
        self.add_output('Pt_out', val=110.0, units='lbf/inch**2', desc='Exit total pressure')
        self.add_output('PR_RDE', val=1.10, desc='Net combustor pressure ratio (Pt_out / Pt_in)')
        self.add_output('Pt_inj', val=88.0, units='lbf/inch**2', desc='Injector plenum total pressure')
        self.add_output('D_cj', val=5500.0, units='ft/s', desc='Chapman-Jouguet detonation wave speed')
        self.add_output('f_rde', val=1750.0, units='Hz', desc='Detonation wave rotational frequency')

        # Analytic derivatives setup
        self.declare_partials('Pt_out', ['Pt_in', 'Tt_in', 'FAR', 'dPqP_inj', 'eta_rde', 'Q_fuel', 'gamma_gas', 'MW_gas'])
        self.declare_partials('PR_RDE', ['Tt_in', 'FAR', 'dPqP_inj', 'eta_rde', 'Q_fuel', 'gamma_gas', 'MW_gas'])
        self.declare_partials('Pt_inj', ['Pt_in', 'dPqP_inj'])
        self.declare_partials('D_cj', ['Tt_in', 'FAR', 'Q_fuel', 'gamma_gas', 'MW_gas'])
        self.declare_partials('f_rde', ['Tt_in', 'FAR', 'Q_fuel', 'gamma_gas', 'MW_gas', 'dia_annulus', 'N_waves'])

    def compute(self, inputs, outputs):
        pt_in = inputs['Pt_in']
        tt_in = inputs['Tt_in']
        far = inputs['FAR']
        dpqp_inj = inputs['dPqP_inj']
        eta = inputs['eta_rde']
        q_fuel = inputs['Q_fuel']
        gamma = inputs['gamma_gas']
        mw = inputs['MW_gas']
        dia = inputs['dia_annulus']
        n_waves = inputs['N_waves']

        # 1. Injector plenum pressure
        u_inj = 1.0 - dpqp_inj
        pt_inj = pt_in * u_inj
        outputs['Pt_inj'] = pt_inj

        # 2. Specific heat release
        q = (far * q_fuel) / (1.0 + far)

        # 3. Gas properties
        r_gas = R_UNIVERSAL_ENG / mw  # Btu/(lbm * degR)

        # 4. Ideal Humphrey pressure ratio
        pi_ideal_minus_1 = (gamma - 1.0) * q / (r_gas * tt_in)

        # 5. Realized pressure gain
        pi_det = 1.0 + eta * pi_ideal_minus_1
        pr_rde = u_inj * pi_det
        pt_out = pt_in * pr_rde

        outputs['Pt_out'] = pt_out
        outputs['PR_RDE'] = pr_rde

        # 6. Wave kinematics
        # 1 Btu = 778.169 ft*lbf, g_c = 32.174
        conv_factor = 778.169 * g_c
        q_mech = q * conv_factor  # (ft*lbf)/lbm
        r_mech = r_gas * conv_factor  # (ft*lbf)/(lbm * degR)
        a_inj = np.sqrt(gamma * r_mech * tt_in)

        xi = (gamma**2 - 1.0) * q_mech / (2.0 * gamma * r_mech * tt_in)
        m_cj = np.sqrt(1.0 + xi) + np.sqrt(xi)
        d_cj = m_cj * a_inj
        outputs['D_cj'] = d_cj

        dia_ft = dia / 12.0
        circ = np.pi * dia_ft
        outputs['f_rde'] = (n_waves * d_cj) / circ

    def compute_partials(self, inputs, J):
        pt_in = inputs['Pt_in']
        tt_in = inputs['Tt_in']
        far = inputs['FAR']
        dpqp_inj = inputs['dPqP_inj']
        eta = inputs['eta_rde']
        q_fuel = inputs['Q_fuel']
        gamma = inputs['gamma_gas']
        mw = inputs['MW_gas']
        dia = inputs['dia_annulus']
        n_waves = inputs['N_waves']

        u_inj = 1.0 - dpqp_inj
        q = (far * q_fuel) / (1.0 + far)
        r_gas = R_UNIVERSAL_ENG / mw
        pi_ideal_minus_1 = (gamma - 1.0) * q / (r_gas * tt_in)
        pi_det = 1.0 + eta * pi_ideal_minus_1
        pr_rde = u_inj * pi_det

        # Pt_inj partials
        J['Pt_inj', 'Pt_in'] = u_inj
        J['Pt_inj', 'dPqP_inj'] = -pt_in

        # PR_RDE partials
        J['PR_RDE', 'dPqP_inj'] = -pi_det
        J['PR_RDE', 'eta_rde'] = u_inj * pi_ideal_minus_1

        # Derivatives of pi_ideal_minus_1:
        # dq/dFAR
        dq_dfar = q_fuel / (1.0 + far)**2
        dpi_dfar = (gamma - 1.0) * dq_dfar / (r_gas * tt_in)
        dpi_dtt = - (gamma - 1.0) * q / (r_gas * tt_in**2)
        dpi_dqfuel = (gamma - 1.0) * far / ((1.0 + far) * r_gas * tt_in)
        dpi_dgamma = q / (r_gas * tt_in)
        dpi_dmw = (gamma - 1.0) * q / (R_UNIVERSAL_ENG * tt_in)

        J['PR_RDE', 'FAR'] = u_inj * eta * dpi_dfar
        J['PR_RDE', 'Tt_in'] = u_inj * eta * dpi_dtt
        J['PR_RDE', 'Q_fuel'] = u_inj * eta * dpi_dqfuel
        J['PR_RDE', 'gamma_gas'] = u_inj * eta * dpi_dgamma
        J['PR_RDE', 'MW_gas'] = u_inj * eta * dpi_dmw

        # Pt_out partials
        J['Pt_out', 'Pt_in'] = pr_rde
        J['Pt_out', 'dPqP_inj'] = pt_in * J['PR_RDE', 'dPqP_inj']
        J['Pt_out', 'eta_rde'] = pt_in * J['PR_RDE', 'eta_rde']
        J['Pt_out', 'FAR'] = pt_in * J['PR_RDE', 'FAR']
        J['Pt_out', 'Tt_in'] = pt_in * J['PR_RDE', 'Tt_in']
        J['Pt_out', 'Q_fuel'] = pt_in * J['PR_RDE', 'Q_fuel']
        J['Pt_out', 'gamma_gas'] = pt_in * J['PR_RDE', 'gamma_gas']
        J['Pt_out', 'MW_gas'] = pt_in * J['PR_RDE', 'MW_gas']

        # D_cj and f_rde partials
        conv_factor = 778.169 * g_c
        q_mech = q * conv_factor
        r_mech = r_gas * conv_factor
        a_inj = np.sqrt(gamma * r_mech * tt_in)

        xi = (gamma**2 - 1.0) * q_mech / (2.0 * gamma * r_mech * tt_in)
        sqrt_1_xi = np.sqrt(1.0 + xi)
        sqrt_xi = np.sqrt(max(xi, 1e-12))
        m_cj = sqrt_1_xi + sqrt_xi
        d_cj = m_cj * a_inj

        # d(m_cj)/d(xi)
        dm_dxi = 0.5 / sqrt_1_xi + 0.5 / sqrt_xi

        # d(xi) derivatives:
        # xi = ((gamma - 1/gamma) * q_mech) / (2.0 * r_mech * tt_in)
        dxi_dq = (gamma**2 - 1.0) * conv_factor / (2.0 * gamma * r_mech * tt_in)
        dxi_dfar = dxi_dq * dq_dfar
        dxi_dqfuel = dxi_dq * (far / (1.0 + far))
        dxi_dtt = - xi / tt_in
        dxi_dmw = xi / mw
        dxi_dgamma = ((2.0 * gamma**2 - (gamma**2 - 1.0)) / (2.0 * gamma**2)) * (q_mech / (r_mech * tt_in))

        # da_inj derivatives:
        da_dtt = 0.5 * a_inj / tt_in
        da_dgamma = 0.5 * a_inj / gamma
        da_dmw = -0.5 * a_inj / mw

        # d(D_cj) = dm_cj * a_inj + m_cj * da_inj
        J['D_cj', 'FAR'] = (dm_dxi * dxi_dfar) * a_inj
        J['D_cj', 'Q_fuel'] = (dm_dxi * dxi_dqfuel) * a_inj
        J['D_cj', 'Tt_in'] = (dm_dxi * dxi_dtt) * a_inj + m_cj * da_dtt
        J['D_cj', 'gamma_gas'] = (dm_dxi * dxi_dgamma) * a_inj + m_cj * da_dgamma
        J['D_cj', 'MW_gas'] = (dm_dxi * dxi_dmw) * a_inj + m_cj * da_dmw

        # f_rde partials
        dia_ft = dia / 12.0
        circ = np.pi * dia_ft
        scale = n_waves / circ

        J['f_rde', 'FAR'] = scale * J['D_cj', 'FAR']
        J['f_rde', 'Q_fuel'] = scale * J['D_cj', 'Q_fuel']
        J['f_rde', 'Tt_in'] = scale * J['D_cj', 'Tt_in']
        J['f_rde', 'gamma_gas'] = scale * J['D_cj', 'gamma_gas']
        J['f_rde', 'MW_gas'] = scale * J['D_cj', 'MW_gas']
        J['f_rde', 'dia_annulus'] = - (n_waves * d_cj) / (np.pi * (dia_ft**2) * 12.0)
        J['f_rde', 'N_waves'] = d_cj / circ
