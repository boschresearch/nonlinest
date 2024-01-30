import examples.ex02_standard_linear as standard_linear
import examples.ex03_parabola as parabola
import examples.ex04_nonlinear as basic_nonlinear
import examples.ex05_nonlinear_2 as particle
import examples.ex06_complete_state_estimation as complete


class TestExamples:
    def test_standard_linear_example(self):
        standard_linear.run_example()

    def test_parabola_example(self):
        parabola.run_example(plot_each_step=False)

    def test_basic_nonlinear(self):
        basic_nonlinear.run_example()

    def test_particle(self):
        particle.run_example()

    def test_complete(self):
        complete.run_example()
