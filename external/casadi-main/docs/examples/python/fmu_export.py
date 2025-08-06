#
#     MIT No Attribution
#
#     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy of this
#     software and associated documentation files (the "Software"), to deal in the Software
#     without restriction, including without limitation the rights to use, copy, modify,
#     merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#     PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#     OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# -*- coding: utf-8 -*-
from casadi import *
from zipfile import ZipFile
from pathlib import Path
import os

# Example how to export FMUs from CasADi
# Joel Andersson, joel@jaeandersson.com

# Start with an empty DaeBuilder instance
dae = DaeBuilder('vdp')

# States
t = dae.add('t', 'independent')
x1 = dae.add('x1', 'output', dict(start = 1))
x2 = dae.add('x2', 'output', dict(start = 0))
q = dae.add('q')

# Input
u = dae.add('u', 'input')

# Set ODE right-hand-sides
dae.eq(dae.der(x1), (1 - x2 * x2)*x1 - x2 + u)
dae.eq(dae.der(x2), x1)
dae.eq(dae.der(q), x1**2 + x2**2 + u**2)

# Add bounds
dae.set_min('u', -0.75)
dae.set_max('u', 1)
dae.set_start('u', 0.5)

# Print DAE
dae.disp(True)

# Export FMU
fmu_files = dae.export_fmu()
print('Generated files: {}'.format(fmu_files))

# Compile DLL
fmi_headers = Path(__file__).parent.parent.parent.parent \
  / 'external_packages' / 'FMI-Standard-3.0' / 'headers'
cfiles = " ".join([f for f in fmu_files if f.endswith('.c')])
sofile = dae.name() + '.so'
os.system(f'gcc --shared -fPIC -I{fmi_headers} {cfiles} -o {sofile}')
print(f'Compiled {sofile}')
fmu_files[sofile] = 'binaries/x86_64-linux'

# Package into an FMU
fmuname = dae.name() + '.fmu'
with ZipFile(fmuname, 'w') as fmufile:
    for f, arcpath in fmu_files.items():
      fmufile.write(f, arcname = arcpath + '/' + f)
      os.remove(f)
print(f'Created FMU: {fmuname}')

# Load the FMU in FMPy
try:
    import fmpy
    fmpy.dump(fmuname)
    # Simulate the generated FMU
    res = fmpy.simulate_fmu(fmuname)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot(res['time'], res['x1'],'-', label = 'x1')
    plt.plot(res['time'], res['x2'],'--', label = 'x2')
    plt.xlabel('time')
    plt.legend()
    plt.grid()
    plt.show()

except ImportError as e:
   print('FMPy not installed. Skipping FMU simulation.')
