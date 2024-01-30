NonlinEst
=========

NonlinEst is a Python library for nonlinear filtering and estimation. It includes many common filtering algorithms such as
* Extended Kalman filter (EKF)
* Unscented Kalman filter (UKF)
* Cubature Kalman filter (CFK)
* Ensemble Kalman filter (EnKF)
* Gaussian Hermite Kalman filter (GHKF)
* Gaussian Particle Filter (GPF)
* Randomized Unscented Kalman Filter (RUKF)
* Sampling Importance Resampling Particle Filter (SIRPF)
* Regularized Particle Filter (RPF)

Installation
------------
To install the library, create a virtual environment and then run

```
pip install .
````
This will automatically install the needed dependencies. If the library should be installed for modification, then use
```
pip install -e .
```
instead.

At least Python 3.6 is required.

Usage
-----
To apply the library to your own problems, you first need to implement your system and measurement model by creating a class that inherits from a suitable sub-class of SystemModel or MeasurementModel.Then you can instantiate one of the provided filters and call the predict() and update() methods.

For example consider the following code snippet

```python
# import library
from nonlinear_estimation_toolbox.filters.ekf import EKF
from nonlinear_estimation_toolbox.distributions import Gaussian
from nonlinear_estimation_toolbox.system_models import AdditiveNoiseSystemModel
from nonlinear_estimation_toolbox.measurement_models import AdditiveNoiseMeasurementModel, MeasurementModel
from nonlinear_estimation_toolbox.utils import ColumnVectors
import numpy as np

# create a filter
filter = EKF()

# set the initial state
initial_state = Gaussian(ColumnVectors([1,2]), np.eye(2))
filter.set_state(initial_state)

# define system model
class SquareSysModel(AdditiveNoiseSystemModel):
    def system_equation(self, state_samples: ColumnVectors) -> ColumnVectors:
        return np.square(state_samples)

# instantiate system model and noise
sys_model = SquareSysModel()
sys_noise = Gaussian(ColumnVectors([0,0]), np.eye(2))
sys_model.set_noise(sys_noise)

# prediction step
filter.predict(sys_model)
print(filter.get_state())

# define measurement model
class CubicMeasModel(AdditiveNoiseMeasurementModel):
    def measurement_equation(self, state_samples: ColumnVectors) -> ColumnVectors:
        return np.power(state_samples, 3)

# instantiate measurement model and noise
meas_model = CubicMeasModel()
meas_noise = Gaussian(ColumnVectors([0,0]), np.eye(2))
meas_model.set_noise(meas_noise)
measurement = ColumnVectors([4,5])

# measurement update step
filter.update(meas_model, measurement)
print(filter.get_state())
```

For further examples, take a look at the provided files in the `examples` subfolder.


Unit Tests
----------
Use `pytest` to run the unit tests.

License
-------

NonlinEst is licenced under the Apache-2.0 license. See the [LICENSE](LICENSE) file for details.

Contributing
------------

Want to contribute? Great! You can do so through the standard GitHub pull request model. For large contributions we encourage you to file a ticket in the GitHub issues tracking system prior to any code development to coordinate with the project development team early in the process. Coordinating up front helps to avoid frustration later on.

Your contribution must be licensed under the Apache-2.0 license, the license used by this project.

Authors
-------
* Florian Faion (email Florian.Faion@de.bosch.com)
* Maxim Dolgov (email Maxim.Dolgov@de.bosch.com)
* Gerhard Kurz (email Gerhard.Kurz2@de.bosch.com)
* Benjamin Schmidt

This toolbox is inspired by ["The Nonlinear Estimation Toolbox"](https://nonlinearestimation.bitbucket.io/), which was developed by Jannik Steinbring.
