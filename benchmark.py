import metrics as m
import romps_heat_index as rhi
import numpy as np
import timeit


if __name__ == "__main__":
    # Realistic AORC data ranges for Texas (1K samples for representative workload)
    # Temperature: 260-320 K (covers winter lows to summer highs)
    temp = np.linspace(290, 320, 1000, dtype=np.float32)
    # Specific humidity: 0.001-0.020 kg/kg (seasonal variation)
    humid = np.linspace(0.001, 0.020, 1000, dtype=np.float32)
    # Pressure: ~101,300 Pa (typical surface pressure)
    pressure = np.full(1000, 101300.0, dtype=np.float32)
    # Wind components: -15 to +15 m/s
    wind_u = np.linspace(-15, 15, 1000, dtype=np.float32)
    wind_v = np.linspace(-15, 15, 1000, dtype=np.float32)

    benchmarks = [
        ("Heat Index", lambda: m.get_aorc_heat_index(temp, humid)),
        ("App. Temp.", lambda: m.get_aorc_apparent_temp(temp, wind_u, wind_v, humid)),
        ("Humidex",    lambda: m.get_aorc_humidex(temp, humid)),
        ("SWBGT",      lambda: m.get_aorc_swbgt(temp, humid)),
        ("WBT",        lambda: m.get_aorc_wbt(temp, humid, pressure)),
        ("RHI",        lambda: rhi.get_aorc_romps_heat_index(temp, humid)),
    ]

    number = 1000   # calls per measurement
    repeat = 5      # number of measurements taken

    for name, func in benchmarks:
        cache_function = func()
        best = min(timeit.repeat(func, number=number, repeat=repeat)) / number
        print(f"{name}: {round(best * 1000, 4)}ms")