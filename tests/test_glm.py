from generate_data import generate_small_rf_and_data
from rfest import GLM
from rfest.utils import uvec, split_data
from rfest.metrics import mse


def test_glm_small_rf():
    w_true, X, y, dims, dt = generate_small_rf_and_data(noise='white')

    df = [3, 4]

    model = GLM(distr='gaussian', output_nonlinearity='none')
    model.add_design_matrix(X, dims=dims, df=df, smooth='cr', filter_nonlinearity='none', name='stimulus')

    model.dt = dt
    model.num_subunits = 1
    model.compute_ci = False

    model.initialize(num_subunits=1, dt=dt, method='random', random_seed=2046)
    model.fit(y=y, num_iters=1000, verbose=100, step_size=0.1, beta=0.01)

    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 1e-1

