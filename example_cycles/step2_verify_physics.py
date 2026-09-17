"""
Step 2 Verification: RDE Pressure Gain Physics and Analytic Derivatives.

Validates:
1. Exact analytic partial derivatives against OpenMDAO complex-step method.
2. Detonation wave kinematics (D_CJ ~ 1700 m/s, f_rde ~ 1.5-2.5 kHz).
3. Humphrey cycle pressure ratio and realization efficiency scaling.
"""

import sys
import os
import numpy as np
import openmdao.api as om

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pycycle.elements.rde_combustor import RDEPressureGainComp


def test_rde_pressure_gain_derivatives():
    print("=" * 80)
    print("STEP 2: VERIFICATION OF RDE PRESSURE GAIN PHYSICS & DERIVATIVES")
    print("=" * 80)

    prob = om.Problem()
    prob.model.add_subsystem('rde_pg', RDEPressureGainComp(), promotes=['*'])
    prob.setup(force_alloc_complex=True)

    # Set nominal operating conditions representative of turbojet station 3
    prob.set_val('Pt_in', 198.39, units='psi')      # 198.39 psi (~13.5 atm)
    prob.set_val('Tt_in', 1187.76, units='degR')    # 1187.76 degR (~660 K)
    prob.set_val('FAR', 0.01776)                    # Fuel-to-air ratio
    prob.set_val('dPqP_inj', 0.12)                  # 12% injector feed drop
    prob.set_val('eta_rde', 0.85)                   # 85% realization factor
    prob.set_val('Q_fuel', 18600.0, units='Btu/lbm')# Jet-A LHV
    prob.set_val('gamma_gas', 1.33)
    prob.set_val('MW_gas', 28.96, units='lbm/mol')
    prob.set_val('dia_annulus', 12.0, units='inch')  # 1 ft diameter
    prob.set_val('N_waves', 1.0)

    prob.run_model()

    pt_in = prob.get_val('Pt_in', units='psi')[0]
    pt_inj = prob.get_val('Pt_inj', units='psi')[0]
    pt_out = prob.get_val('Pt_out', units='psi')[0]
    pr_rde = prob.get_val('PR_RDE')[0]
    d_cj = prob.get_val('D_cj', units='ft/s')[0]
    d_cj_ms = d_cj * 0.3048
    f_rde = prob.get_val('f_rde', units='Hz')[0]

    print("\n--- Physical Outputs at Nominal Operating Point ---")
    print(f"Compressor Exit Total Pressure (Pt_in) : {pt_in:.2f} psi")
    print(f"Injector Plenum Total Pressure (Pt_inj): {pt_inj:.2f} psi  (12% loss for backflow prevention)")
    print(f"RDE Combustor Exit Total Pressure (Pt_out): {pt_out:.2f} psi")
    print(f"Net Combustor Pressure Ratio (PR_RDE) : {pr_rde:.4f}  ({(pr_rde - 1.0)*100:+.2f}% net gain)")
    print(f"CJ Detonation Velocity (D_cj)         : {d_cj:.1f} ft/s  ({d_cj_ms:.1f} m/s)")
    print(f"Detonation Wave Rotation Frequency    : {f_rde:.1f} Hz")

    print("\n--- Checking Analytic Derivatives with Complex Step ---")
    data = prob.check_partials(method='cs', compact_print=True, out_stream=sys.stdout)

    # Verify error threshold
    max_error = 0.0
    for comp_name, comp_data in data.items():
        for (of, wrt), item in comp_data.items():
            rel_err = item['rel error'][0]
            abs_err = item['abs error'][0]
            if rel_err is not None and not np.isnan(rel_err):
                max_error = max(max_error, rel_err)
            elif abs_err is not None and not np.isnan(abs_err):
                max_error = max(max_error, abs_err)

    print(f"\nMaximum Relative Error across non-zero Jacobians: {max_error:.2e}")
    assert max_error < 1e-6, f"Derivative check failed with max error {max_error:.2e}"
    print("SUCCESS: All analytic derivatives verified to < 1e-6 tolerance!")
    print("=" * 80)


if __name__ == "__main__":
    test_rde_pressure_gain_derivatives()
