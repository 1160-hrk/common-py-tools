# 物理定数などを定義するファイル
from scipy.constants import *

PLANCK_CONSTANT = h
PLANCK_CONSTANT_REDUCED = hbar
BOLTZMANN_CONSTANT = k
AVOGADRO_CONSTANT = N_A
GAS_CONSTANT = R
SPEED_OF_LIGHT = c
ELECTRON_MASS = m_e
PROTON_MASS = m_p
NEUTRON_MASS = m_n
ELEMENTARY_CHARGE = e
PERMITTIVITY_VACUUM = epsilon_0
PERMEABILITY_VACUUM = mu_0
GRAVITATIONAL_CONSTANT = G
BOLTZMANN_CONSTANT_JOULE = k
BOLTZMANN_CONSTANT_EV = k * electron_volt
# 物理定数の単位を定義する
# SI単位系
SI_UNIT = {
    "length": "m",
    "mass": "kg",
    "time": "s",
    "electric_current": "A",
    "temperature": "K",
    "amount_of_substance": "mol",
    "luminous_intensity": "cd",
}
# CGS単位系
CGS_UNIT = {
    "length": "cm",
    "mass": "g",
    "time": "s",
    "electric_current": "A",
    "temperature": "K",
    "amount_of_substance": "mol",
    "luminous_intensity": "cd",
}
# 電子ボルト
EV_UNIT = {
    "energy": "eV",
    "charge": "e",
}
# エネルギー
ENERGY_UNIT = {
    "joule": "J",
    "electron_volt": "eV",
    "calorie": "cal",
    "kilowatt_hour": "kWh",
}
# 圧力
PRESSURE_UNIT = {
    "pascal": "Pa",
    "bar": "bar",
    "atmosphere": "atm",
    "torr": "torr",
}
# 温度
TEMPERATURE_UNIT = {
    "kelvin": "K",
    "celsius": "°C",
    "fahrenheit": "°F",
}
# 角度
ANGLE_UNIT = {
    "radian": "rad",
    "degree": "°",
    "gradian": "g",
}
# 角速度
ANGULAR_VELOCITY_UNIT = {
    "radian_per_second": "rad/s",
    "degree_per_second": "°/s",
    "revolution_per_minute": "rpm",
}
# 角加速度
ANGULAR_ACCELERATION_UNIT = {
    "radian_per_second_squared": "rad/s²",
    "degree_per_second_squared": "°/s²",
    "revolution_per_minute_squared": "rpm/s",
}
# 速度
SPEED_UNIT = {
    "meter_per_second": "m/s",
    "kilometer_per_hour": "km/h",
    "mile_per_hour": "mph",
}
# 加速度
ACCELERATION_UNIT = {
    "meter_per_second_squared": "m/s²",
    "kilometer_per_hour_squared": "km/h²",
    "mile_per_hour_squared": "mph/s",
}
# 力
FORCE_UNIT = {
    "newton": "N",
    "kilogram_force": "kgf",
    "pound_force": "lbf",
}
# エネルギー
ENERGY_UNIT = {
    "joule": "J",
    "kilojoule": "kJ",
    "calorie": "cal",
    "kilocalorie": "kcal",
    "electron_volt": "eV",
}
# 仕事
WORK_UNIT = {
    "joule": "J",
    "kilojoule": "kJ",
    "calorie": "cal",
    "kilocalorie": "kcal",
    "electron_volt": "eV",
}
# 熱量
HEAT_UNIT = {
    "joule": "J",
    "kilojoule": "kJ",
    "calorie": "cal",
    "kilocalorie": "kcal",
    "electron_volt": "eV",
}
# 電力

