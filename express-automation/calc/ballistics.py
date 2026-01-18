import subprocess
import pyperclip
import time
import pyautogui

import domain
from calc.const import app_file, app_dir
from calc.misc import click, field_type, locate_with_cache


# calculate_ballistics calculates values using Express software
def calculate_ballistics(target: domain.TargetInfo, la: domain.LAInfo,
                         integration: domain.IntegrationInfo,
                         calculate_aerodynamics: bool) -> domain.BallisticsCalculationResult:
    # Open Express
    proc = subprocess.Popen(
        [app_file],
        cwd=app_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

    # Open "Ballistic calculation" menu
    click("./assets/menu/main-calculation.png")
    time.sleep(0.1)
    click("./assets/menu/main-ballistic.png")
    time.sleep(0.3)

    # Type Target values
    field_type("./assets/target/v.png", f"{target.velocity:.0f}".replace(".", ","))
    field_type("./assets/target/theta.png", f"{target.theta:.6f}".replace(".", ","))
    field_type("./assets/target/x.png", f"{target.x:.2f}".replace(".", ","))
    field_type("./assets/target/y.png", f"{target.y:.0f}".replace(".", ","))

    # Switch to LA section
    click("./assets/la/btn.png")
    time.sleep(0.25)

    # Type LA values
    field_type("./assets/la/m_0.png", f"{la.m_0:.0f}".replace(".", ","))
    field_type("./assets/la/d_middle.png", f"{la.d_middle:.1f}".replace(".", ","))

    field_type("./assets/la/mdot_A.png", f"{la.mdot_A:.1f}".replace(".", ","))
    field_type("./assets/la/mdot_B.png", f"{la.mdot_B:.10f}".replace(".", ","))

    field_type("./assets/la/t_engine.png", f"{la.t_engine:.6f}".replace(".", ","))
    field_type("./assets/la/d_nozzle.png", f"{la.d_nozzle:.2f}".replace(".", ","))
    field_type("./assets/la/p_nozzle.png", f"{la.p_nozzle:.0f}".replace(".", ","))
    field_type("./assets/la/J.png", f"{la.J:.0f}".replace(".", ","))

    field_type("./assets/la/V_0.png", f"{la.V_0:.0f}".replace(".", ","))
    field_type("./assets/la/theta_0.png", f"{la.theta_0:.6f}".replace(".", ","))

    # Switch to Aerodynamics section
    click("./assets/aerodynamics/btn.png")
    time.sleep(0.5)
    click("./assets/aerodynamics/file-btn.png")
    time.sleep(0.1)
    click("./assets/aerodynamics/open-btn.png")
    time.sleep(0.2)
    pyautogui.typewrite(la.aerodynamics, interval=0.005)
    pyautogui.press("enter")
    time.sleep(0.1)
    if calculate_aerodynamics:
        click("./assets/aerodynamics/calculation-btn.png")
        time.sleep(0.1)
        click("./assets/aerodynamics/calculation-confirm.png")
        time.sleep(0.3)
    pyautogui.hotkey("alt" + "f4")
    click("./assets/aerodynamics/close.png")

    # Select Guidance (Only three-point for now)
    click("./assets/guidance/btn.png")
    time.sleep(0.1)
    click("./assets/guidance/3-points.png")

    # Switch to Integration params
    click("./assets/integration/btn.png")
    time.sleep(0.2)
    click("./assets/integration/btn-2.png")
    time.sleep(0.1)
    field_type("./assets/integration/step.png", f"{integration.step:.2f}".replace(".", ","))
    field_type("./assets/integration/output-step.png", f"{integration.output_step:.0f}".replace(".", ","))

    # Calculate
    click("./assets/menu/ballistic-calculation.png")
    time.sleep(0.1)
    click("./assets/menu/ballistic-calculation-proceed.png")
    time.sleep(0.25)

    # View results
    click("./assets/menu/ballistic-calculation.png")
    time.sleep(0.2)
    click("./assets/menu/ballistic-calculation-results.png")
    time.sleep(0.2)

    # Results/Forces
    click("./assets/menu/results-view.png")
    time.sleep(0.1)
    click("./assets/menu/results-view-forces.png")
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "c")

    forces_text = pyperclip.paste()
    forces = forces_text.split("\n")[-1].strip().split()
    m_final = float(forces[1])
    mdot_final = float(forces[2])

    # Results/Location
    click("./assets/menu/results-view.png")
    time.sleep(0.1)
    click("./assets/menu/results-view-location.png")
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "c")

    location_text = pyperclip.paste()
    location = location_text.split("\n")[-1].strip().split()
    v_final = float(location[1])
    theta_final = float(location[2])
    x_final = float(location[5])

    # Results/Additional
    click("./assets/menu/results-view.png")
    time.sleep(0.1)
    click("./assets/menu/results-view-additional.png")
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "c")
    additional_text = pyperclip.paste()

    proc.terminate()

    return domain.BallisticsCalculationResult(m_final, mdot_final, v_final, theta_final, x_final, forces_text,
                                              location_text, additional_text)
