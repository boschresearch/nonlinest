from nonlinear_estimation_toolbox.filters.gpf import GPF
from nonlinear_estimation_toolbox.measurement_models import Likelihood, MeasurementModel
from nonlinear_estimation_toolbox.system_models import SystemModel
from nonlinear_estimation_toolbox.utils import ColumnVectors


class CGPF(GPF):
    """The Gaussian particle filter with a combined time and measurement update.

    This class implements the step() method (combined predict and update)
    to execute the auxiliary SIRPF. The update and predict functions can still
    be called separately and equal to the functions from SIRPF.
    """

    def _perform_step(
        self,
        sys_model: SystemModel,
        meas_model: MeasurementModel,
        measurement: ColumnVectors,
    ) -> None:
        particles = self.get_state().draw_random_samples(self.num_particles)
        predicted_particles = self._predict_particles(
            sys_model, particles, self.num_particles
        )

        if isinstance(meas_model, Likelihood):
            # Predicted state estimate is not Gaussian, cannot use decomposed state update
            updated_state_mean, updated_state_cov = self._update_likelihood(
                meas_model, measurement, predicted_particles
            )
            self._check_and_save_update(updated_state_mean, updated_state_cov)
        else:
            raise TypeError("meas_model must be of type Likelihood")
